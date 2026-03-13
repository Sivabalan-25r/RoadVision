"""
RoadVision — Plate Reader (YOLO Detection + CRNN Recognition)
Detects license plates using a dedicated YOLOv8 plate detector model,
applies geometric filters, preprocesses crops, and reads text via CRNN.
"""

import logging
import os
import re
from typing import Optional, Tuple, List

import cv2
import numpy as np

from recognition.crnn_recognizer import recognize_plate

logger = logging.getLogger(__name__)

# ---- YOLO Plate Detector ----
_plate_model = None

PLATE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'license_plate_detector.pt'
)

DETECTION_CONF = 0.5  # YOLO detection confidence threshold (raised to reduce FPs)

# ---- Geometric Filter Thresholds ----
MIN_ASPECT_RATIO = 1.5   # w/h minimum (Indian two-wheeler plates can be squarer)
MAX_ASPECT_RATIO = 6.0   # w/h maximum
MIN_PLATE_WIDTH = 60     # Minimum crop width in pixels (smaller plates on two-wheelers)
MIN_PLATE_HEIGHT = 18    # Minimum crop height in pixels
MAX_WIDTH_RATIO = 0.5    # Max plate width relative to frame width (stricter)
MIN_Y_RATIO = 0.30       # Reject plates in top 30% of frame (Indian dashcams mounted lower)

# ---- Text Cleaning ----
MIN_CLEANED_LENGTH = 6
MAX_CLEANED_LENGTH = 12

OCR_CORRECTIONS = {
    '|': '1', 'l': '1', 'i': '1',
    'o': '0', 'q': '0',
    's': '5', 'z': '2', 'b': '8', 'g': '6',
    '(': 'C', ')': 'J', '[': 'L', ']': 'J',
}

KNOWN_FALSE_POSITIVES = {
    'IND', 'INDIA', 'TEST', 'SAMPLE', 'DEMO',
    'GOVT', 'POLICE', 'TAXI', 'AUTO',
}


def _get_plate_model():
    """Load the YOLO model for license plate detection.

    Prefers the dedicated license_plate_detector.pt model.
    Falls back to yolov8n.pt (general object detection) if the plate
    detector is not available — geometric filters will compensate.
    """
    global _plate_model
    if _plate_model is None:
        from ultralytics import YOLO

        if os.path.exists(PLATE_MODEL_PATH):
            logger.info(f"Loading YOLO plate detector: {PLATE_MODEL_PATH}")
            _plate_model = YOLO(PLATE_MODEL_PATH)
        else:
            # Fallback to general YOLO model — geometric filters will
            # reject windshield/bumper false positives
            fallback = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'yolov8s.pt'
            )
            if not os.path.exists(fallback):
                fallback = 'yolov8s.pt'
            logger.warning(
                f"Plate detector not found at {PLATE_MODEL_PATH}. "
                f"Falling back to general model: {fallback}. "
                f"For best results, add license_plate_detector.pt."
            )
            _plate_model = YOLO(fallback)

    return _plate_model


# ---- Geometric Filtering ----

def passes_geometric_filter(
    x1: int, y1: int, x2: int, y2: int,
    frame_width: int, frame_height: int,
) -> bool:
    """
    Apply geometric filters to a detected bounding box.

    Rejects if:
      - Aspect ratio (w/h) outside [2, 6]
      - Width < 80 or height < 25
      - Width > 50% of frame width
      - Top edge (y1) in upper 45% of frame
    """
    w = x2 - x1
    h = y2 - y1

    if h <= 0 or w <= 0:
        return False

    ratio = w / h
    if ratio < MIN_ASPECT_RATIO or ratio > MAX_ASPECT_RATIO:
        return False

    if w < MIN_PLATE_WIDTH or h < MIN_PLATE_HEIGHT:
        return False

    if w > frame_width * MAX_WIDTH_RATIO:
        return False

    if y1 < frame_height * MIN_Y_RATIO:
        return False

    return True


# ---- Plate Preprocessing ----

def preprocess_plate_crop(plate_crop: np.ndarray) -> np.ndarray:
    """
    Preprocess a plate crop for CRNN recognition.

    Steps:
      1. Convert to grayscale
      2. Bilateral filter (noise reduction, edge preservation)
      3. OTSU threshold (binarization)
    """
    # Grayscale
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()

    # Bilateral filter: reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # OTSU threshold
    _, thresh = cv2.threshold(
        filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


# ---- YOLO Detection ----

# COCO class IDs for vehicles (used when falling back to yolov8n.pt)
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Track whether we're using a dedicated plate detector or general model
_is_plate_detector = None


def _check_is_plate_detector():
    """Check if the loaded model is a dedicated plate detector or general model."""
    global _is_plate_detector
    if _is_plate_detector is None:
        _is_plate_detector = os.path.exists(PLATE_MODEL_PATH)
    return _is_plate_detector


def _estimate_plate_region(x1: int, y1: int, x2: int, y2: int,
                           frame_shape: tuple) -> tuple:
    """
    Estimate the license plate region from a vehicle bounding box.
    Plates are typically at the bottom-center of the vehicle.

    Returns: (px1, py1, px2, py2) of estimated plate region.
    """
    vw = x2 - x1
    vh = y2 - y1

    # Plate is roughly in the bottom 20% of the vehicle, center 40%
    px1 = int(x1 + vw * 0.25)
    py1 = int(y1 + vh * 0.75)
    px2 = int(x1 + vw * 0.75)
    py2 = int(y1 + vh * 0.95)

    # Clamp to frame bounds
    h_max, w_max = frame_shape[:2]
    px1 = max(0, min(px1, w_max - 1))
    py1 = max(0, min(py1, h_max - 1))
    px2 = max(px1 + 10, min(px2, w_max))
    py2 = max(py1 + 5, min(py2, h_max))

    return px1, py1, px2, py2


def detect_plates(
    frame: np.ndarray,
) -> List[dict]:
    """
    Detect license plates in a single frame.

    If using a dedicated plate detector: bounding boxes are plate regions.
    If using yolov8n fallback: detects vehicles and estimates plate regions.

    Returns list of dicts with:
      - 'bbox': [x, y, w, h]
      - 'crop': preprocessed plate image (numpy)
      - 'raw_crop': original BGR plate crop
      - 'confidence': YOLO detection confidence
    """
    model = _get_plate_model()
    is_plate_model = _check_is_plate_detector()
    frame_height, frame_width = frame.shape[:2]

    results = model(frame, verbose=False, conf=DETECTION_CONF)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if is_plate_model:
                # Dedicated plate detector — bbox IS the plate
                plate_x1, plate_y1, plate_x2, plate_y2 = x1, y1, x2, y2
            else:
                # General model (yolov8n) — only process vehicles
                if cls_id not in VEHICLE_CLASSES:
                    continue
                # Estimate plate region from vehicle bbox
                plate_x1, plate_y1, plate_x2, plate_y2 = \
                    _estimate_plate_region(x1, y1, x2, y2, frame.shape)

            # Apply geometric filters on the plate region
            if not passes_geometric_filter(
                plate_x1, plate_y1, plate_x2, plate_y2,
                frame_width, frame_height
            ):
                # For vehicle fallback, relax filters (plate estimate may be small)
                if not is_plate_model:
                    pw = plate_x2 - plate_x1
                    ph = plate_y2 - plate_y1
                    if pw < 40 or ph < 10:
                        continue
                    # Accept the estimated region even if filters fail
                else:
                    continue

            # Crop the plate region
            raw_crop = frame[plate_y1:plate_y2, plate_x1:plate_x2]
            if raw_crop.size == 0:
                continue

            # Preprocess for recognition
            processed = preprocess_plate_crop(raw_crop)

            w = plate_x2 - plate_x1
            h = plate_y2 - plate_y1
            detections.append({
                'bbox': [plate_x1, plate_y1, w, h],
                'crop': processed,
                'raw_crop': raw_crop,
                'confidence': conf,
            })

    return detections


# ---- Text Cleaning & Validation ----

def clean_text(text: str) -> str:
    """
    Clean raw OCR/CRNN text for plate processing.

    Steps:
      1. Uppercase
      2. Apply character corrections
      3. Remove whitespace, hyphens, dots
      4. Remove non-alphanumeric characters
    """
    text = text.strip().upper()

    corrected = []
    for ch in text:
        lower = ch.lower()
        if lower in OCR_CORRECTIONS:
            corrected.append(OCR_CORRECTIONS[lower])
        else:
            corrected.append(ch)
    text = ''.join(corrected)

    text = re.sub(r'[\s\-\.\,\;\:\'\"\`]+', '', text)
    text = re.sub(r'[^A-Z0-9]', '', text)

    return text


def is_garbage_text(text: str) -> bool:
    """
    Detect garbage recognition results that are clearly not number plates.
    Returns True if the text should be discarded.
    """
    if len(text) < MIN_CLEANED_LENGTH:
        return True

    if len(text) > MAX_CLEANED_LENGTH:
        return True

    if text in KNOWN_FALSE_POSITIVES:
        return True

    # All same character
    if len(set(text)) <= 1:
        return True

    # All digits or all letters
    if text.isdigit() or text.isalpha():
        return True

    # Repeating 2-char patterns
    if len(text) >= 6:
        pair = text[:2]
        if text == pair * (len(text) // 2) + pair[:len(text) % 2]:
            return True

    # Low entropy
    unique_ratio = len(set(text)) / len(text)
    if unique_ratio < 0.3 and len(text) >= 6:
        return True

    # Must start with a letter (Indian plates start with state code)
    if not text[0].isalpha():
        return True

    # Must have at least 2 letters and 2 digits
    letter_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    if letter_count < 2 or digit_count < 2:
        return True

    return False


def normalize_plate(text: str) -> str:
    """Lightweight normalizer for deduplication."""
    text = text.upper().strip()
    text = re.sub(r'[\s\-\.\,]+', '', text)
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


# ---- Public API ----

def read_plate(plate_image: np.ndarray) -> Optional[str]:
    """
    Read text from a preprocessed plate crop image using CRNN.

    Pipeline:
      1. CRNN recognition → raw text + confidence
      2. Confidence filter (≥ 0.3)
      3. Clean text
      4. Garbage rejection

    Args:
        plate_image: Preprocessed grayscale/thresholded plate crop.

    Returns:
        Cleaned plate text or None if unreadable/garbage.
    """
    if plate_image is None or plate_image.size == 0:
        return None

    try:
        raw_text, confidence = recognize_plate(plate_image)

        if not raw_text or confidence < 0.3:
            logger.debug(f"CRNN result too low confidence: '{raw_text}' ({confidence:.2f})")
            return None

        cleaned = clean_text(raw_text)

        if is_garbage_text(cleaned):
            logger.debug(f"Rejected garbage CRNN result: '{cleaned}'")
            return None

        return cleaned

    except Exception as e:
        logger.error(f"Plate read error: {e}")
        return None


def get_read_confidence(plate_image: np.ndarray) -> float:
    """
    Get the CRNN recognition confidence for a plate image.

    Returns:
        Float between 0 and 1. Returns 0.0 on failure.
    """
    if plate_image is None or plate_image.size == 0:
        return 0.0

    try:
        _, confidence = recognize_plate(plate_image)
        return confidence
    except Exception as e:
        logger.error(f"CRNN confidence error: {e}")
        return 0.0
