"""
RoadVision — Plate Reader (YOLOv8 Dedicated License Plate Detector)
Loads the keremberke/yolov8-license-plate model once at startup,
detects plates, applies geometric filters, preprocesses crops for OCR,
and reads text via PaddleOCR (or CRNN when weights are available).
"""

import logging
import os
import re
import sys
from typing import Optional, Tuple, List, Any, Dict, Union, cast

# Ensure backend-python is in path for local imports
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.append(_root)

import cv2      # type: ignore
import numpy as np  # type: ignore

try:
    from recognition.crnn_recognizer import recognize_plate # type: ignore
except ImportError:
    # Fallback for linter/environment issues
    def recognize_plate(p, v): return ("", 0.0)

logger = logging.getLogger(__name__)

# ---- YOLO Plate Detector ----
_plate_model = None

PLATE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'license_plate_detector.pt'
)

# Aggressive filtering to eliminate background ghosts
DETECTION_CONF = 0.65 

# ---- Geometric Filter Thresholds (tuned for Indian roads) ----
MIN_ASPECT_RATIO = 2.0   # w/h minimum (Indian plates are wide)
MAX_ASPECT_RATIO = 5.0   # w/h maximum
MIN_PLATE_AREA = 3000    # Minimum pixels
MAX_PLATE_AREA = 100000  # Reject anything taking too much screen (likely error)

# ---- OCR Confidence Threshold ----
OCR_MIN_CONFIDENCE = 0.5 

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

    from ultralytics import YOLO # type: ignore
    import torch                # type: ignore
    import functools

    if not os.path.exists(PLATE_MODEL_PATH):
        raise FileNotFoundError(
            f"License plate detector model not found at: {PLATE_MODEL_PATH}\n"
            f"Please place 'license_plate_detector.pt' in backend-python/models/.\n"
            f"This model is required — the server cannot start without it."
        )

    logger.info(f"Loading YOLOv8 plate detector: {PLATE_MODEL_PATH}")
    
    # PyTorch 2.6+ defaults to weights_only=True which breaks older YOLO models.
    # Temporarily patch torch.load to use weights_only=False for model loading.
    _original_torch_load = torch.load
    torch.load = functools.partial(_original_torch_load, weights_only=False)
    try:
        _plate_model = YOLO(PLATE_MODEL_PATH)
    finally:
        torch.load = _original_torch_load
    
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
      - Aspect ratio (w/h) outside [2.0, 5.0]
      - Area < 3000 pixels
    """
    w = x2 - x1
    h = y2 - y1

    if h <= 0 or w <= 0:
        return False

    area = w * h
    if area < MIN_PLATE_AREA or area > MAX_PLATE_AREA:
        return False

    ratio = w / h
    if ratio < MIN_ASPECT_RATIO or ratio > MAX_ASPECT_RATIO:
        return False

    return True


# ---- Plate Preprocessing ----
def preprocess_plate_variants(plate_crop: np.ndarray) -> List[np.ndarray]:
    """
    Generate multiple variants of a plate crop for ensemble OCR.
    
    Returns:
        List of [Original, CLAHE-only, Sharpened, Thresholded]
    """
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()

    # Base upscale
    h, w = gray.shape
    if w < 320 or h < 120:
        gray = cv2.resize(gray, (320, 120), interpolation=cv2.INTER_CUBIC)
    
    variants = []
    
    # Variant 1: Original Grayscale (Contrast normalized)
    variants.append(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX))
    
    # Variant 2: CLAHE only (Good for varying light)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))
    
    # Variant 3: Sharpened + CLAHE
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    variants.append(clahe.apply(sharpened))
    
    # Variant 4: Adaptive Threshold (Binary)
    thresh = cv2.adaptiveThreshold(
        variants[1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    variants.append(thresh)
    
    return variants


def preprocess_plate_crop(plate_crop: np.ndarray) -> np.ndarray:
    """Legacy wrapper for single-crop logic."""
    variants = preprocess_plate_variants(plate_crop)
    return variants[2] # Return the sharpened one as default


# ---- YOLO Detection ----

def detect_plates(
    frame: np.ndarray,
    frame_number: int = 0,
) -> List[Dict[str, Any]]:
    """
    Detect license plates in a single frame using the dedicated
    YOLOv8 license plate detector model.

    Returns list of dicts with:
      - 'bbox': [x, y, w, h]
      - 'crop': preprocessed plate image (numpy, 320×120)
      - 'raw_crop': original BGR plate crop
      - 'confidence': YOLO detection confidence
    """
    model: Any = _get_plate_model()
    results: Any = model(frame, verbose=False, conf=DETECTION_CONF)
    detections: List[Dict[str, Any]] = []
    
    # Initialize with definite types for linter
    filtered_count: int = 0
    detection_idx: int = 0

    # Get class name mapping from model
    class_names: Dict[int, str] = getattr(model, 'names', {}) or {}

    # Create debug directory
    debug_dir = str(os.path.join(os.path.dirname(__file__), '..', 'debug_plates'))
    os.makedirs(debug_dir, exist_ok=True)

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Only keep 'license_plate' class detections
            if isinstance(class_names, dict) and cls_id in class_names:
                cls_name = str(class_names[cls_id]).lower()
            else:
                cls_name = ""
            cls_name = re.sub(r"[^a-z0-9]", "", cls_name)
            if cls_name not in ('licenseplate', 'plate', ''):
                if class_names:
                    continue

            # map(int, ...) ensures they are clearly integers for indexing
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Clamp coordinates to frame boundaries
            h_img, w_img = frame.shape[:2]
            clamped_x1 = max(0, x1)
            clamped_y1 = max(0, y1)
            clamped_x2 = min(w_img, x2)
            clamped_y2 = min(h_img, y2)

            # Apply geometric filters
            if not passes_geometric_filter(clamped_x1, clamped_y1, clamped_x2, clamped_y2):
                filtered_count = filtered_count + 1 # type: ignore
                continue

            # Crop the plate region
            # Cast frame to Any or use # type: ignore to bypass strict slicing lints
            frame_arr: Any = frame
            raw_crop = frame_arr[clamped_y1:clamped_y2, clamped_x1:clamped_x2] # type: ignore
            if raw_crop.size == 0:
                continue

            # Save debug image
            debug_path = os.path.join(debug_dir, f'frame{frame_number}_plate{detection_idx}_raw.png')
            cv2.imwrite(debug_path, raw_crop)

            # Preprocess for recognition (upscale → CLAHE → blur → Sharpen)
            processed = preprocess_plate_crop(raw_crop)

            # Save preprocessed debug image
            debug_path_proc = os.path.join(debug_dir, f'frame{frame_number}_plate{detection_idx}_processed.png')
            cv2.imwrite(debug_path_proc, processed)

            w_box = clamped_x2 - clamped_x1
            h_box = clamped_y2 - clamped_y1
            detections.append({
                'bbox': [clamped_x1, clamped_y1, w_box, h_box],
                'crop': processed,
                'raw_crop': raw_crop,
                'confidence': conf,
            })
            detection_idx = detection_idx + 1 # type: ignore

    if filtered_count > 0:
        logger.debug(f"Filtered {filtered_count} plates by geometric constraints")

    logger.info(f"Frame {frame_number}: Detected {len(detections)} valid plates")
    return detections


# ---- Text Cleaning & Validation ----

def clean_text(text: str) -> str:
    """
    Clean raw OCR text for plate processing.
    """
    text = text.strip().upper()

    corrected = [
        OCR_CORRECTIONS.get(ch.lower(), ch)
        for ch in text
    ]
    text = ''.join(corrected)

    text = re.sub(r'[\s\-\.\,\;\:\'\"\`]+', '', text)
    text = re.sub(r'[^A-Z0-9]', '', text)

    return text


def is_garbage_text(text: str) -> bool:
    """
    Detect garbage recognition results.
    """
    if len(text) < MIN_CLEANED_LENGTH or len(text) > MAX_CLEANED_LENGTH:
        return True

    if text in KNOWN_FALSE_POSITIVES:
        return True

    # All same character
    if len(set(text)) <= 1:
        return True

    # Repeating 2-char patterns (e.g., "ABABAB")
    if len(text) >= 6:
        if re.match(r"^(..)\1+$", str(text)):
            return True

    # Low entropy
    unique_ratio = len(set(text)) / len(text)
    if unique_ratio < 0.25 and len(text) >= 6:
        return True

    # Must have at least 2 letters (state code) and 2 digits
    letter_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    if letter_count < 2 or digit_count < 2:
        return True

    # No single character should repeat more than 4 times
    from collections import Counter
    counts = Counter(text)
    if counts and max(counts.values()) > 4:
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
    Read text from a plate crop image using an Ensemble OCR strategy.
    """
    if plate_image is None or plate_image.size == 0:
        return None

    # Generate image variants
    variants = preprocess_plate_variants(plate_image)
    
    candidates = []
    
    # Run OCR on each variant
    for variant in variants:
        try:
            raw_text, confidence = recognize_plate(plate_image, variant)
            if not raw_text or confidence < OCR_MIN_CONFIDENCE:
                continue
                
            cleaned = clean_text(raw_text)
            if is_garbage_text(cleaned):
                continue
            
            # Validation
            try:
                from rules import plate_rules # type: ignore
                validation = plate_rules.validate_plate(cleaned)
            except ImportError:
                validation = None
            
            # Score
            score = confidence
            if validation and not getattr(validation, "violation", True):
                score += 1.0 
                
            candidates.append({
                'text': cleaned,
                'score': score,
                'confidence': confidence
            })
        except Exception as e:
            logger.error(f"Ensemble variant error: {e}")

    if not candidates:
        return None
        
    # Pick best
    candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
    best = candidates[0]
    
    logger.info(f"Ensemble Pick: '{best.get('text')}'")
    return cast(str, best.get('text'))


def get_read_confidence(plate_image: np.ndarray, preprocessed_image: np.ndarray = None) -> float:
    """
    Get the OCR recognition confidence for a plate image.
    """
    if plate_image is None or plate_image.size == 0:
        return 0.0

    try:
        ocr_image = preprocessed_image if preprocessed_image is not None else plate_image
        _, confidence = recognize_plate(plate_image, ocr_image)
        return float(confidence)
    except Exception as e:
        logger.error(f"OCR confidence error: {e}")
        return 0.0
