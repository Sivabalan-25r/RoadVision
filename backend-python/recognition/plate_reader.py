"""
RoadVision — Plate Reader (YOLOv26 Dedicated License Plate Detector)
Loads the YOLOv26 license plate detector model once at startup,
detects plates, applies geometric filters, preprocesses crops for OCR,
and reads text via PaddleOCR (or CRNN when weights are available).
"""

import logging
import os
import re
import sys
import time
from typing import Optional, Tuple, List, Any, Dict, Union, cast

# Ensure backend-python is in path for local imports
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.append(_root)

import cv2      # type: ignore
import numpy as np  # type: ignore

try:
    from ultralytics import YOLO # type: ignore
    import torch                # type: ignore
except ImportError:
    pass

try:
    from recognition.crnn_recognizer import recognize_plate # type: ignore
except ImportError:
    # Fallback for linter/environment issues
    def recognize_plate(p, v): return ("", 0.0)

# Import centralized configuration
import config

logger = logging.getLogger(__name__)

# ---- YOLO Plate Detector ----
_plate_model = None
_use_half_precision = False  # Will be set during model loading

# Configuration values are now imported from config.py
# Access via config.MIN_ASPECT_RATIO, config.YOLO_CONFIDENCE_THRESHOLD, etc.


def load_plate_model():
    """Load the dedicated YOLOv26 license plate detector model.

    MUST be called once at startup. Raises FileNotFoundError if the
    model file is missing — the server cannot operate without it.
    
    Logs model parameters, inference configuration, and performance metrics.
    """
    global _plate_model, _use_half_precision
    if _plate_model is not None:
        return _plate_model

    import functools
    import time

    if not os.path.exists(config.YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"License plate detector model not found at: {config.YOLO_MODEL_PATH}\n"
            f"Please place 'license_plate_detector.pt' in backend-python/models/.\n"
            f"This model is required — the server cannot start without it."
        )

    logger.info(f"Loading YOLOv26 plate detector: {config.YOLO_MODEL_PATH}")
    
    # PyTorch 2.6+ defaults to weights_only=True which breaks older YOLO models.
    # Temporarily patch torch.load to use weights_only=False for model loading.
    _original_torch_load = torch.load
    torch.load = functools.partial(_original_torch_load, weights_only=False)
    
    load_start = time.time()
    try:
        _plate_model = YOLO(config.YOLO_MODEL_PATH)
    finally:
        torch.load = _original_torch_load
    
    load_time = time.time() - load_start
    
    # Auto-detect CUDA and enable FP16 half-precision if available
    cuda_available = False
    try:
        if torch.cuda.is_available():
            cuda_available = True
            _use_half_precision = config.YOLO_USE_HALF_PRECISION
            if _use_half_precision:
                logger.info(f"  ✓ CUDA GPU detected — enabling FP16 half-precision for speed")
            else:
                logger.info(f"  ✓ CUDA GPU detected — FP16 disabled in config")
        else:
            _use_half_precision = False
            logger.info(f"  ℹ CUDA not available — using CPU inference")
    except Exception as e:
        logger.warning(f"  ⚠ CUDA detection failed: {e}")
        _use_half_precision = False
    
    # Log model parameters and configuration
    try:
        # Get model parameter count
        param_count = sum(p.numel() for p in _plate_model.model.parameters())
        param_count_m = param_count / 1_000_000
        
        logger.info(f"  ✓ YOLOv26 plate detector loaded successfully")
        logger.info(f"    - Model parameters: {param_count_m:.1f}M")
        logger.info(f"    - Load time: {load_time:.2f}s")
        logger.info(f"    - Inference size: {config.YOLO_IMAGE_SIZE}×{config.YOLO_IMAGE_SIZE}")
        logger.info(f"    - Confidence threshold: {config.YOLO_CONFIDENCE_THRESHOLD}")
        logger.info(f"    - FP16 half-precision: {'Enabled' if _use_half_precision else 'Disabled'}")
        logger.info(f"    - Device: {'CUDA GPU' if cuda_available else 'CPU'}")
    except Exception as e:
        logger.warning(f"  ⚠ Could not retrieve model parameters: {e}")
        logger.info(f"  ✓ YOLOv26 plate detector loaded successfully")
    
    # Run a quick inference test to measure performance
    try:
        test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        inference_start = time.time()
        _ = _plate_model(
            test_frame,
            verbose=False,
            conf=config.YOLO_CONFIDENCE_THRESHOLD,
            imgsz=config.YOLO_IMAGE_SIZE,
            half=_use_half_precision,
        )
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms
        logger.info(f"    - Test inference time: {inference_time:.1f}ms")
    except Exception as e:
        logger.warning(f"  ⚠ Test inference failed: {e}")
    
    return _plate_model


def _get_plate_model():
    """Return the already-loaded YOLO model (singleton)."""
    global _plate_model
    if _plate_model is None:
        _plate_model = load_plate_model()
    
    return _plate_model


# ---- Geometric Filtering ----

def passes_geometric_filter(
    x1: int, y1: int, x2: int, y2: int,
    frame_width: int = 0, frame_height: int = 0,
) -> bool:
    """
    Apply geometric filters to a detected plate bounding box.

    Rejects if:
      - Aspect ratio (w/h) outside [1.5, 7.0]
      - Width < 50 or height < 15
      - Area < 750 pixels
      - Area > 15% of frame (rejects windshield/window detections)
    """
    w = x2 - x1
    h = y2 - y1

    if h <= 0 or w <= 0:
        return False

    area = w * h
    if area < config.MIN_PLATE_AREA or area > config.MAX_PLATE_AREA:
        return False

    ratio = w / h
    if ratio < config.MIN_ASPECT_RATIO or ratio > config.MAX_ASPECT_RATIO:
        return False
    
    # Reject detections that are too large (e.g., windshield, full vehicle)
    if frame_width > 0 and frame_height > 0:
        frame_area = frame_width * frame_height
        if area > frame_area * config.MAX_PLATE_AREA_RATIO:
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


def _detect_noise_level(image: np.ndarray) -> float:
    """
    Detect noise level in an image using Laplacian variance.
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def _apply_multiple_thresholds(image: np.ndarray) -> np.ndarray:
    """
    Apply multiple threshold methods and select the best result.
    """
    # Method 1: Adaptive Gaussian
    thresh_gaussian = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Method 2: Adaptive Mean
    thresh_mean = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Method 3: OTSU
    _, thresh_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    def edge_density(img):
        edges = cv2.Canny(img, 50, 150)
        return np.sum(edges > 0) / img.size
    
    candidates = [
        (thresh_gaussian, edge_density(thresh_gaussian)),
        (thresh_mean, edge_density(thresh_mean)),
        (thresh_otsu, edge_density(thresh_otsu))
    ]
    
    best_thresh, _ = max(candidates, key=lambda x: x[1])
    return best_thresh


def preprocess_plate_crop(plate_crop: np.ndarray) -> np.ndarray:
    """
    Preprocess a plate crop for OCR recognition with adaptive pipeline.
    """
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()

    # Aggressive upscaling
    h, w = gray.shape
    target_height = 160
    
    if h < target_height:
        scale = target_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = gray.shape
    
    # Bilateral filtering (the "Secret Sauce" from RoadVision Deep Dive)
    # Smooths noise while keeping character edges sharp — faster than NLM denoising
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # CLAHE (Adaptive Histogram Equalization) — fixes shadows and glares
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    thresh = _apply_multiple_thresholds(sharpened)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    target_width = 400
    aspect_ratio = w / h if h > 0 else 1
    target_height = int(target_width / aspect_ratio)
    target_height = max(80, min(200, target_height))
    
    return cv2.resize(thresh, (target_width, target_height), interpolation=cv2.INTER_CUBIC)



# ---- Bounding Box Refinement ----

def _refine_bounding_box(
    plate_crop: np.ndarray,
    original_bbox: Tuple[int, int, int, int]
) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """
    Apply light padding to YOLO bounding box to remove vehicle body.
    
    Only trims a small border — does NOT use contour detection,
    which was previously cutting off the last digits of plates.
    
    Args:
        plate_crop: Original BGR plate crop from YOLO detection
        original_bbox: Original (x1, y1, x2, y2) coordinates
        
    Returns:
        Tuple of (refined_crop, refined_bbox)
    """
    if plate_crop is None or plate_crop.size == 0:
        return None, original_bbox
    
    h, w = plate_crop.shape[:2]
    x1_orig, y1_orig, x2_orig, y2_orig = original_bbox
    
    # Only trim 2% from edges — enough to remove border noise
    # without cutting off characters at the edges
    padding_percent = 0.02
    pad_x = int(w * padding_percent)
    pad_y = int(h * padding_percent)
    
    # Safety: never crop more than a few pixels
    pad_x = min(pad_x, 5)
    pad_y = min(pad_y, 3)
    
    # Ensure we don't over-crop tiny plates
    if pad_x * 2 >= w or pad_y * 2 >= h:
        return plate_crop, original_bbox
    
    cropped = plate_crop[pad_y:h-pad_y, pad_x:w-pad_x]
    
    if cropped.size == 0:
        return plate_crop, original_bbox
    
    new_x1 = x1_orig + pad_x
    new_y1 = y1_orig + pad_y
    new_x2 = x2_orig - pad_x
    new_y2 = y2_orig - pad_y
    
    return cropped, (new_x1, new_y1, new_x2, new_y2)


# ---- YOLO Detection ----

def detect_plates(
    frame: np.ndarray,
    frame_number: int = 0,
) -> List[Dict[str, Any]]:
    """
    Detect license plates in a single frame using the dedicated
    YOLOv26 license plate detector model.

    Returns list of dicts with:
      - 'bbox': [x, y, w, h]
      - 'crop': preprocessed plate image (numpy, 320×120)
      - 'raw_crop': original BGR plate crop
      - 'confidence': YOLO detection confidence
    """
    model = _get_plate_model()
    results = model(
        frame,
        verbose=False,
        conf=config.YOLO_CONFIDENCE_THRESHOLD,
        imgsz=config.YOLO_IMAGE_SIZE,
        half=_use_half_precision,
        agnostic_nms=True,
    )
    detections: List[Dict[str, Any]] = []
    filtered_count: int = 0
    detection_idx: int = 0

    # Get class name mapping from model
    class_names: Dict[int, str] = getattr(model, 'names', {}) or {}


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

            # Apply geometric filters (pass frame dimensions for max area check)
            if not passes_geometric_filter(clamped_x1, clamped_y1, clamped_x2, clamped_y2, w_img, h_img):
                filtered_count += 1 # type: ignore
                w, h = clamped_x2 - clamped_x1, clamped_y2 - clamped_y1
                ratio = w / h if h > 0 else 0
                logger.debug(
                    f"Filtered plate: size={w}x{h}, ratio={ratio:.2f}, "
                    f"conf={conf:.2f}"
                )
                continue

            # Crop the plate region
            # Cast frame to Any or use # type: ignore to bypass strict slicing lints
            frame_arr: Any = frame
            raw_crop = frame_arr[clamped_y1:clamped_y2, clamped_x1:clamped_x2] # type: ignore
            if raw_crop.size == 0:
                continue

            # Apply tight bounding box refinement to remove vehicle body/background
            refined_crop, refined_bbox = _refine_bounding_box(raw_crop, (x1, y1, x2, y2))
            if refined_crop is None or refined_crop.size == 0:
                logger.debug(f"Bounding box refinement failed for detection {detection_idx}")
                continue
            
            # Update coordinates with refined bbox
            x1, y1, x2, y2 = refined_bbox

            # Preprocess for recognition (upscale → CLAHE → blur → OTSU → 320×120)
            processed = preprocess_plate_crop(refined_crop)

            w_box = clamped_x2 - clamped_x1
            h_box = clamped_y2 - clamped_y1
            detections.append({
                'bbox': [clamped_x1, clamped_y1, w_box, h_box],
                'crop': processed,
                'raw_crop': refined_crop,
                'confidence': conf,
            })
            detection_idx += 1 # type: ignore

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
        config.OCR_CORRECTIONS.get(ch.lower(), ch)
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
    if len(text) < config.MIN_PLATE_LENGTH or len(text) > config.MAX_PLATE_LENGTH:
        return True

    if text in config.KNOWN_FALSE_POSITIVES:
        return True

    # All same character
    if len(set(text)) <= 1:
        return True

    # Repeating 2-char patterns (e.g., "ABABAB")
    if len(text) >= 6:
        if re.match(r"^(..)\1+$", str(text)):
            return True

    # Low entropy (made more lenient)
    unique_ratio = len(set(text)) / len(text)
    if unique_ratio < 0.20 and len(text) >= 8:  # Only reject if very low entropy AND long
        return True

    # Must have at least 1 letter and 1 digit
    letter_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    if letter_count < 1 or digit_count < 1:
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

def _validate_indian_plate_format(text: str) -> bool:
    """
    Validate that OCR result matches Indian plate format.
    
    Expected format: AA NN AA NNNN
    - AA: State code (2 letters)
    - NN: District code (2 digits)
    - AA: Series letters (1-3 letters)
    - NNNN: Registration number (1-4 digits)
    
    Total length: 6-11 characters (min: AA NN A N, max: AA NN AAA NNNN)
    
    Returns:
        True if text matches expected format, False otherwise.
    """
    if not text or len(text) < 6 or len(text) > 12:  # Relaxed max length
        return False
    
    # Relaxed Indian plate pattern: at least 2 letters, 2 digits, then letters and digits
    # This catches more variations while still filtering garbage
    pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z0-9]{2,8}$')
    return bool(pattern.match(text))


def _apply_position_based_corrections(text: str) -> str:
    """
    Apply character corrections based on expected position in Indian plate format.
    
    Format: AA NN AA NNNN
    - Positions 0-1: MUST be letters (state code)
    - Positions 2-3: MUST be digits (district code)
    - Positions 4+: Series (letters) followed by registration number (digits)
    
    Returns:
        Corrected text with position-based character substitutions.
    """
    if len(text) < 6:
        return text
    
    corrected = list(text)
    
    # Positions 0-1: State code (MUST be letters)
    for i in range(min(2, len(text))):
        ch = text[i]
        if ch.isdigit():
            # Digit in letter position - try to convert
            digit_to_letter = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G', '4': 'A', '7': 'T'}
            if ch in digit_to_letter:
                corrected[i] = digit_to_letter[ch]
    
    # Positions 2-3: District code (MUST be digits)
    for i in range(2, min(4, len(text))):
        ch = text[i]
        if ch.isalpha():
            # Letter in digit position - try to convert
            letter_to_digit = {'O': '0', 'I': '1', 'L': '1', 'B': '8', 'S': '5', 'Z': '2', 'G': '6', 'D': '0', 'Q': '0', 'T': '7', 'A': '4'}
            if ch in letter_to_digit:
                corrected[i] = letter_to_digit[ch]
    
    # Positions 4+: Series (1-3 letters) followed by registration number (1-4 digits)
    if len(text) > 4:
        suffix = text[4:] # type: ignore
        
        # Find where series ends and number begins
        # Strategy: Look for the first sequence of consecutive digits
        # Everything before that is series, everything after is registration number
        series_end = len(suffix)  # Default: all series (no digits found)
        
        for j in range(len(suffix)):
            ch_suffix = str(suffix[j])
            if ch_suffix.isdigit():
                # Found first digit - this is where registration number starts
                series_end = j
                break
        
        # Clamp series length to max 3 letters
        if series_end > 3:
            series_end = 3
        
        # Series positions: MUST be letters
        for j in range(series_end):
            idx = 4 + j
            if idx < len(text):
                ch = text[idx]
                ch_str = str(ch)
                if ch_str.isdigit():
                    digit_to_letter = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G', '4': 'A', '7': 'T'}
                    if ch_str in digit_to_letter:
                        corrected[idx] = digit_to_letter[ch_str]
        
        # Registration number positions: MUST be digits
        for j in range(series_end, len(suffix)):
            idx = 4 + j
            if idx < len(text):
                ch = text[idx]
                ch_str = str(ch)
                if ch_str.isalpha():
                    letter_to_digit = {'O': '0', 'I': '1', 'L': '1', 'B': '8', 'S': '5', 'Z': '2', 'G': '6', 'D': '0', 'Q': '0', 'T': '7', 'A': '4'}
                    if ch_str in letter_to_digit:
                        corrected[idx] = letter_to_digit[ch_str]
    
    return ''.join(corrected)


def read_plate(plate_image: np.ndarray, preprocessed_image: np.ndarray = None) -> Optional[str]:
    """
    Read text from a plate crop image using an Ensemble OCR strategy.
    
    Pipeline:
      1. Run OCR on multiple image variants
      2. Scoring and validation
      3. Return best candidate
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
            if not raw_text or confidence < config.OCR_CONFIDENCE_THRESHOLD:
                continue
                
            cleaned = clean_text(raw_text)
            
            # Apply position-based corrections
            cleaned = _apply_position_based_corrections(cleaned)
            
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
        return float(confidence) if confidence is not None else 0.0
    except Exception as e:
        logger.error(f"OCR confidence error: {e}")
        return 0.0
