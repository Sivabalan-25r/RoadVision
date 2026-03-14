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

DETECTION_CONF = 0.25  # YOLO detection confidence threshold (lowered to catch more plates)

# ---- Geometric Filter Thresholds (tuned for Indian roads) ----
MIN_ASPECT_RATIO = 1.2   # w/h minimum (standard plate ratio)
MAX_ASPECT_RATIO = 7.0   # w/h maximum (wide plates)
MIN_PLATE_WIDTH = 40     # Minimum crop width in pixels (lowered to catch more plates)
MIN_PLATE_HEIGHT = 12    # Minimum crop height in pixels (lowered to catch more plates)
MIN_PLATE_AREA = 480     # Minimum area in pixels (lowered to catch more plates)

# ---- OCR Confidence Threshold ----
OCR_MIN_CONFIDENCE = 0.25  # Lowered to catch more valid plates

# ---- Text Cleaning ----
MIN_CLEANED_LENGTH = 5  # Reduced from 6 to catch shorter plates
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
    import torch
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
      - Aspect ratio (w/h) outside [1.5, 6.0]
      - Width < 100 or height < 30
      - Area < 3000 pixels
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
    
    area = w * h
    if area < MIN_PLATE_AREA:
        return False

    return True


# ---- Plate Preprocessing ----

def _detect_noise_level(image: np.ndarray) -> float:
    """
    Detect noise level in an image using Laplacian variance.
    
    Returns:
        Float representing noise level (higher = more noise).
        Threshold: < 100 = noisy, >= 100 = clean
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def _apply_multiple_thresholds(image: np.ndarray) -> np.ndarray:
    """
    Apply multiple threshold methods and select the best result.
    
    Tries:
      1. Adaptive Gaussian threshold
      2. Adaptive Mean threshold
      3. OTSU threshold
      
    Selects the result with the highest edge density (most character edges).
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
    
    # Evaluate each method by edge density (more edges = better character separation)
    def edge_density(img):
        edges = cv2.Canny(img, 50, 150)
        return np.sum(edges > 0) / img.size
    
    candidates = [
        (thresh_gaussian, edge_density(thresh_gaussian)),
        (thresh_mean, edge_density(thresh_mean)),
        (thresh_otsu, edge_density(thresh_otsu))
    ]
    
    # Select the threshold with highest edge density
    best_thresh, _ = max(candidates, key=lambda x: x[1])
    return best_thresh


def preprocess_plate_crop(plate_crop: np.ndarray) -> np.ndarray:
    """
    Preprocess a plate crop for OCR recognition with adaptive pipeline.

    Adaptive improvements:
      1. Convert to grayscale
      2. Aggressive upscaling for small plates
      3. Advanced denoising
      4. CLAHE for contrast enhancement
      5. Sharpening filter
      6. Multiple threshold methods with best result selection
      7. Morphological operations to clean up characters
      8. Resize to OCR-friendly size (preserves aspect ratio)
    """
    # Grayscale
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()

    # Aggressive upscaling - scale to at least 160px height for better OCR
    h, w = gray.shape
    target_height = 160  # Increased from 120 for better character recognition
    
    if h < target_height:
        # Calculate scale to reach target height while preserving aspect ratio
        scale = target_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = gray.shape
    
    # Advanced denoising - always apply for better results
    # Use Non-local Means Denoising for better quality
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # CLAHE for contrast enhancement with stronger settings
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply sharpening filter to enhance character edges
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

    # Apply multiple threshold methods and select best result
    thresh = _apply_multiple_thresholds(sharpened)
    
    # Morphological operations to clean up and connect broken characters
    # Use a small kernel to close gaps in characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Resize to standard OCR input size while preserving aspect ratio
    # Target: 400 width (increased from 320), adjust height to maintain aspect ratio
    target_width = 400
    aspect_ratio = w / h
    target_height = int(target_width / aspect_ratio)
    
    # Clamp height to reasonable range for OCR (80-200 pixels)
    target_height = max(80, min(200, target_height))
    
    plate_processed = cv2.resize(thresh, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    return plate_processed


# ---- Bounding Box Refinement ----

def _refine_bounding_box(
    plate_crop: np.ndarray,
    original_bbox: Tuple[int, int, int, int]
) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """
    Refine YOLO bounding box to tightly crop only the plate region.
    
    Removes vehicle body and background by:
      1. Cropping 5-10% padding from edges
      2. Applying edge detection to find plate boundaries
      3. Validating the refined region contains plate-like content
    
    Args:
        plate_crop: Original BGR plate crop from YOLO detection
        original_bbox: Original (x1, y1, x2, y2) coordinates
        
    Returns:
        Tuple of (refined_crop, refined_bbox) or (None, original_bbox) if refinement fails
    """
    if plate_crop is None or plate_crop.size == 0:
        return None, original_bbox
    
    h, w = plate_crop.shape[:2]
    x1_orig, y1_orig, x2_orig, y2_orig = original_bbox
    
    # Step 1: Crop 3-5% padding from edges to remove vehicle body/background
    # Use 4% as a balanced middle ground (reduced from 7% to preserve more plate content)
    padding_percent = 0.04
    pad_x = int(w * padding_percent)
    pad_y = int(h * padding_percent)
    
    # Ensure we don't over-crop
    if pad_x * 2 >= w or pad_y * 2 >= h:
        pad_x = max(1, w // 20)
        pad_y = max(1, h // 20)
    
    cropped = plate_crop[pad_y:h-pad_y, pad_x:w-pad_x]
    
    if cropped.size == 0:
        return None, original_bbox
    
    # Step 2: Apply edge detection to find actual plate boundaries
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped
    
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # No edges found, use the padded crop
        new_x1 = x1_orig + pad_x
        new_y1 = y1_orig + pad_y
        new_x2 = x2_orig - pad_x
        new_y2 = y2_orig - pad_y
        return cropped, (new_x1, new_y1, new_x2, new_y2)
    
    # Find the bounding rectangle that encompasses all significant contours
    # Filter contours by area to ignore noise
    min_contour_area = (cropped.shape[0] * cropped.shape[1]) * 0.01  # 1% of image
    significant_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    if not significant_contours:
        # Use all contours if filtering removed everything
        significant_contours = contours
    
    # Get bounding box of all significant contours
    all_points = np.vstack(significant_contours)
    x, y, w_box, h_box = cv2.boundingRect(all_points)
    
    # Step 3: Validate the refined region
    # Ensure the refined box is reasonable (not too small)
    if w_box < cropped.shape[1] * 0.3 or h_box < cropped.shape[0] * 0.3:
        # Refined box is too small, use padded crop instead
        new_x1 = x1_orig + pad_x
        new_y1 = y1_orig + pad_y
        new_x2 = x2_orig - pad_x
        new_y2 = y2_orig - pad_y
        return cropped, (new_x1, new_y1, new_x2, new_y2)
    
    # Extract the refined crop
    refined = cropped[y:y+h_box, x:x+w_box]
    
    # Calculate new absolute coordinates
    new_x1 = x1_orig + pad_x + x
    new_y1 = y1_orig + pad_y + y
    new_x2 = new_x1 + w_box
    new_y2 = new_y1 + h_box
    
    return refined, (new_x1, new_y1, new_x2, new_y2)


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

            # Apply tight bounding box refinement to remove vehicle body/background
            refined_crop, refined_bbox = _refine_bounding_box(raw_crop, (x1, y1, x2, y2))
            if refined_crop is None or refined_crop.size == 0:
                logger.debug(f"Bounding box refinement failed for detection {detection_idx}")
                continue
            
            # Update coordinates with refined bbox
            x1, y1, x2, y2 = refined_bbox

            # Save debug image
            debug_path = os.path.join(debug_dir, f'frame{frame_number}_plate{detection_idx}_raw.png')
            cv2.imwrite(debug_path, refined_crop)
            logger.info(f"Saved debug plate: {debug_path} (size: {refined_crop.shape})")

            # Preprocess for recognition (upscale → CLAHE → blur → OTSU → 320×120)
            processed = preprocess_plate_crop(refined_crop)

            # Save preprocessed debug image
            debug_path_proc = os.path.join(debug_dir, f'frame{frame_number}_plate{detection_idx}_processed.png')
            cv2.imwrite(debug_path_proc, processed)

            w = x2 - x1
            h = y2 - y1
            detections.append({
                'bbox': [x1, y1, w, h],
                'crop': processed,
                'raw_crop': refined_crop,
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

    # Low entropy (made more lenient)
    unique_ratio = len(set(text)) / len(text)
    if unique_ratio < 0.20 and len(text) >= 8:  # Only reject if very low entropy AND long
        return True

    # Must have at least 1 letter and 1 digit (relaxed from 2 letters)
    letter_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    if letter_count < 1 or digit_count < 1:
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
        suffix = text[4:]
        
        # Find where series ends and number begins
        # Strategy: Look for the first sequence of consecutive digits
        # Everything before that is series, everything after is registration number
        series_end = len(suffix)  # Default: all series (no digits found)
        
        for j in range(len(suffix)):
            if suffix[j].isdigit():
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
                if ch.isdigit():
                    digit_to_letter = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G', '4': 'A', '7': 'T'}
                    if ch in digit_to_letter:
                        corrected[idx] = digit_to_letter[ch]
        
        # Registration number positions: MUST be digits
        for j in range(series_end, len(suffix)):
            idx = 4 + j
            if idx < len(text):
                ch = text[idx]
                if ch.isalpha():
                    letter_to_digit = {'O': '0', 'I': '1', 'L': '1', 'B': '8', 'S': '5', 'Z': '2', 'G': '6', 'D': '0', 'Q': '0', 'T': '7', 'A': '4'}
                    if ch in letter_to_digit:
                        corrected[idx] = letter_to_digit[ch]
    
    return ''.join(corrected)


def read_plate(plate_image: np.ndarray, preprocessed_image: np.ndarray = None) -> Optional[str]:
    """
    Read text from a plate crop image using CRNN/PaddleOCR/Tesseract.

    Pipeline:
      1. OCR recognition → raw text + confidence
      2. Confidence filter (≥ 0.25)
      3. Clean text
      4. Garbage rejection
      5. Return text (validation is lenient to show more results)

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

        # Very lenient validation - just check basic structure
        if len(cleaned) >= 6 and len(cleaned) <= 12:
            # Has at least 2 letters and 1 digit? Good enough!
            letter_count = sum(1 for c in cleaned if c.isalpha())
            digit_count = sum(1 for c in cleaned if c.isdigit())
            
            if letter_count >= 2 and digit_count >= 1:
                logger.info(f"Accepted plate: '{cleaned}'")
                return cleaned
        
        logger.debug(f"Rejected: basic validation failed: '{cleaned}'")
        return None

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
