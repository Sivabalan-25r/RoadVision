"""
RoadVision — Plate Reader (YOLOv8 Dedicated License Plate Detector)
Loads the keremberke/yolov8-license-plate model once at startup,
detects plates, applies geometric filters, preprocesses crops for OCR,
and reads text via PaddleOCR (or CRNN when weights are available).
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

DETECTION_CONF = 0.3  # YOLO detection confidence threshold (lowered to catch more plates)

# ---- Geometric Filter Thresholds (tuned for Indian roads) ----
MIN_ASPECT_RATIO = 1.2   # w/h minimum (more lenient for two-wheeler plates)
MAX_ASPECT_RATIO = 7.0   # w/h maximum (more lenient for wide plates)
MIN_PLATE_WIDTH = 80     # Minimum crop width in pixels (increased for OCR quality)
MIN_PLATE_HEIGHT = 25    # Minimum crop height in pixels (increased for OCR quality)

# ---- OCR Confidence Threshold ----
OCR_MIN_CONFIDENCE = 0.4  # Lowered to catch more plates (was too strict)

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


def load_plate_model():
    """Load the dedicated YOLOv8 license plate detector model.

    MUST be called once at startup. Raises FileNotFoundError if the
    model file is missing — the server cannot operate without it.
    """
    global _plate_model
    if _plate_model is not None:
        return _plate_model

    from ultralytics import YOLO

    if not os.path.exists(PLATE_MODEL_PATH):
        raise FileNotFoundError(
            f"License plate detector model not found at: {PLATE_MODEL_PATH}\n"
            f"Please place 'license_plate_detector.pt' in backend-python/models/.\n"
            f"This model is required — the server cannot start without it."
        )

    logger.info(f"Loading YOLOv8 plate detector: {PLATE_MODEL_PATH}")
    _plate_model = YOLO(PLATE_MODEL_PATH)
    logger.info("YOLOv8 plate detector loaded successfully.")
    return _plate_model


def _get_plate_model():
    """Return the already-loaded YOLO model (singleton)."""
    global _plate_model
    if _plate_model is None:
        return load_plate_model()
    return _plate_model


# ---- Geometric Filtering ----

def passes_geometric_filter(
    x1: int, y1: int, x2: int, y2: int,
) -> bool:
    """
    Apply geometric filters to a detected plate bounding box.

    Rejects if:
      - Aspect ratio (w/h) outside [1.5, 6.0]
      - Width < 60 or height < 18
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

    return True


# ---- Plate Preprocessing ----

def preprocess_plate_crop(plate_crop: np.ndarray) -> np.ndarray:
    """
    Preprocess a plate crop for OCR recognition with ANPR-optimized pipeline.

    Steps:
      1. Convert to grayscale
      2. Upscale 3x for better character resolution
      3. CLAHE for contrast enhancement
      4. Slight Gaussian blur to reduce noise
      5. OTSU threshold (binarization)
      6. Resize to larger OCR input size (320×120 instead of 160×40)
    """
    # Grayscale
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()

    # Upscale 3x before processing to preserve character details
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # CLAHE for contrast enhancement (better than bilateral filter)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Slight Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # OTSU threshold
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Resize to larger OCR input size (320×120 instead of 160×40)
    # This preserves more character detail
    plate_processed = cv2.resize(thresh, (320, 120), interpolation=cv2.INTER_CUBIC)

    return plate_processed


# ---- YOLO Detection ----

def detect_plates(
    frame: np.ndarray,
    frame_number: int = 0,
) -> List[dict]:
    """
    Detect license plates in a single frame using the dedicated
    YOLOv8 license plate detector model.

    Only keeps detections with class 'license_plate' (or class index 0
    for single-class plate detector models).

    Returns list of dicts with:
      - 'bbox': [x, y, w, h]
      - 'crop': preprocessed plate image (numpy, 320×120)
      - 'raw_crop': original BGR plate crop
      - 'confidence': YOLO detection confidence
    """
    model = _get_plate_model()

    results = model(frame, verbose=False, conf=DETECTION_CONF)
    detections = []
    filtered_count = 0

    # Get class name mapping from model
    class_names = model.names if hasattr(model, 'names') else {}

    # Create debug directory
    import os
    debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_plates')
    os.makedirs(debug_dir, exist_ok=True)

    detection_idx = 0
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Only keep 'license_plate' class detections
            cls_name = class_names.get(cls_id, '').lower()
            cls_name = re.sub(r"[^a-z0-9]", "", cls_name)
            if cls_name not in ('licenseplate', 'plate', ''):
                if class_names:
                    continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Clamp coordinates to frame boundaries
            h_img, w_img = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)

            # Apply geometric filters
            if not passes_geometric_filter(x1, y1, x2, y2):
                filtered_count += 1
                w, h = x2 - x1, y2 - y1
                ratio = w / h if h > 0 else 0
                logger.debug(
                    f"Filtered plate: size={w}x{h}, ratio={ratio:.2f}, "
                    f"conf={conf:.2f}"
                )
                continue

            # Crop the plate region
            raw_crop = frame[y1:y2, x1:x2]
            if raw_crop.size == 0:
                continue

            # Save debug image
            debug_path = os.path.join(debug_dir, f'frame{frame_number}_plate{detection_idx}_raw.png')
            cv2.imwrite(debug_path, raw_crop)
            logger.info(f"Saved debug plate: {debug_path} (size: {raw_crop.shape})")

            # Preprocess for recognition (upscale → CLAHE → blur → OTSU → 320×120)
            processed = preprocess_plate_crop(raw_crop)

            # Save preprocessed debug image
            debug_path_proc = os.path.join(debug_dir, f'frame{frame_number}_plate{detection_idx}_processed.png')
            cv2.imwrite(debug_path_proc, processed)

            w = x2 - x1
            h = y2 - y1
            detections.append({
                'bbox': [x1, y1, w, h],
                'crop': processed,
                'raw_crop': raw_crop,
                'confidence': conf,
            })
            detection_idx += 1

    if filtered_count > 0:
        logger.debug(f"Filtered {filtered_count} plates by geometric constraints")

    logger.info(f"Frame {frame_number}: Detected {len(detections)} valid plates")
    return detections


# ---- Text Cleaning & Validation ----

def clean_text(text: str) -> str:
    """
    Clean raw OCR text for plate processing.

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

    # Repeating 2-char patterns
    if len(text) >= 6:
        pair = text[:2]
        if text == pair * (len(text) // 2) + pair[:len(text) % 2]:
            return True

    # Low entropy
    unique_ratio = len(set(text)) / len(text)
    if unique_ratio < 0.25 and len(text) >= 6:
        return True

    # Must start with a letter (Indian plates start with state code)
    if not text[0].isalpha():
        return True

    # Must have at least 2 letters and 1 digit (more lenient)
    letter_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    if letter_count < 2 or digit_count < 1:
        return True

    return False


def normalize_plate(text: str) -> str:
    """Lightweight normalizer for deduplication."""
    text = text.upper().strip()
    text = re.sub(r'[\s\-\.\,]+', '', text)
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


# ---- Public API ----

def read_plate(plate_image: np.ndarray, preprocessed_image: np.ndarray = None) -> Optional[str]:
    """
    Read text from a plate crop image using CRNN/PaddleOCR/Tesseract.

    Pipeline:
      1. OCR recognition → raw text + confidence
      2. Confidence filter (≥ 0.4)
      3. Clean text
      4. Garbage rejection

    Args:
        plate_image: Raw BGR plate crop (for PaddleOCR).
        preprocessed_image: Preprocessed grayscale plate (for Tesseract).

    Returns:
        Cleaned plate text or None if unreadable/garbage.
    """
    if plate_image is None or plate_image.size == 0:
        return None

    try:
        # Use preprocessed image for Tesseract if available
        ocr_image = preprocessed_image if preprocessed_image is not None else plate_image
        raw_text, confidence = recognize_plate(plate_image, ocr_image)

        if not raw_text:
            logger.debug("OCR returned empty text")
            return None
            
        if confidence < OCR_MIN_CONFIDENCE:
            logger.debug(f"OCR confidence too low: '{raw_text}' ({confidence:.2f})")
            return None

        cleaned = clean_text(raw_text)
        logger.info(f"OCR: '{raw_text}' → cleaned: '{cleaned}' (conf: {confidence:.2f})")

        if is_garbage_text(cleaned):
            logger.debug(f"Rejected as garbage: '{cleaned}'")
            return None

        return cleaned

    except Exception as e:
        logger.error(f"Plate read error: {e}")
        return None


def get_read_confidence(plate_image: np.ndarray, preprocessed_image: np.ndarray = None) -> float:
    """
    Get the OCR recognition confidence for a plate image.

    Returns:
        Float between 0 and 1. Returns 0.0 on failure.
    """
    if plate_image is None or plate_image.size == 0:
        return 0.0

    try:
        ocr_image = preprocessed_image if preprocessed_image is not None else plate_image
        _, confidence = recognize_plate(plate_image, ocr_image)
        return confidence
    except Exception as e:
        logger.error(f"OCR confidence error: {e}")
        return 0.0
