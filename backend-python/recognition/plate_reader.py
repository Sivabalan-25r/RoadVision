"""
EvasionEye — Plate Reader (YOLOv26 Dedicated License Plate Detector)
Loads the YOLOv26 license plate detector model once at startup,
detects plates, applies geometric filters, preprocesses crops for OCR,
and reads text via PaddleOCR (or CRNN when weights are available).
"""

import logging
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Any, Dict, Union, cast

# Ensure backend-python is in path for local imports
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.append(_root)

import cv2      # type: ignore
import numpy as np  # type: ignore

# Disable PaddleOCR slow connectivity check (avoids 10s delay + false ImportError)
import os as _os
_os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

try:
    from ultralytics import YOLO # type: ignore
    import torch                # type: ignore
except ImportError:
    pass

try:
    from paddleocr import PaddleOCR as _PaddleOCR  # type: ignore
    _PADDLEOCR_AVAILABLE = True
except ImportError:
    _PaddleOCR = None  # type: ignore
    _PADDLEOCR_AVAILABLE = False


# Import centralized configuration
import config
from recognition.bayesian_arbitrator import BayesianOCRArbitrator
from stabilization.kalman_tracker import TrackerManager

logger = logging.getLogger(__name__)

# ---- Bayesian Arbitrator ----
_arbitrator = BayesianOCRArbitrator(threshold=getattr(config, 'BAYESIAN_OCR_THRESHOLD', 0.65))

# ---- Kalman Tracker Manager ----
_tracker_manager = TrackerManager(
    max_unseen=getattr(config, 'KALMAN_SKIP_FRAMES', 3),
    max_cov=getattr(config, 'KALMAN_MAX_COVARIANCE', 5.0)
)


# ---- YOLO Plate Detector ----
_plate_model = None
_use_half_precision = False  # Will be set during model loading

# ---- PaddleOCR Instance ----
_PADDLE_UNAVAILABLE = object()  # Sentinel: means we tried and it's not available
_paddleocr_instance = None
_paddle_lock = threading.Lock()  # Prevents 4-thread race on first load

# ---- EasyOCR Instance ----
_easyocr_instance = None
_easyocr_lock = threading.Lock()

# ---- Tesseract availability flag ----
_tesseract_available: Optional[bool] = None

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
            f"Please place 'yolov26-license-plate.pt' in backend-python/models/.\n"
            f"This model is required — the server cannot start without it."
        )

    logger.info(f"Loading YOLOv26 plate detector: {config.YOLO_MODEL_PATH}")
    
    # PyTorch 2.6+ defaults to weights_only=True which breaks older YOLO models.
    # Temporarily patch torch.load to use weights_only=False for model loading.
    _original_torch_load = torch.load
    torch.load = functools.partial(_original_torch_load, weights_only=False)
    
    load_start = time.time()
    try:
        # Initialize model
        _plate_model = YOLO(config.YOLO_MODEL_PATH)
        
        # YOLOv26 Optimization: Set device once
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"

        # Check if model is NMS-Free (YOLO26 feature)
        has_nms = any('nms' in str(m).lower() for m in _plate_model.model.modules())

        # Enable FP16 half-precision if CUDA available for YOLOv26 real-time speed
        if device == "cuda":
            _plate_model.model.half() # FP16 half-precision for real-time GPU performance
            _use_half_precision = True
            logger.info(f"  ✓ CUDA GPU detected — YOLOv26 FP16 half-precision ENABLED")
        else:
            _use_half_precision = False
            logger.info(f"  ℹ Using {device.upper()} inference mode")

        _plate_model.to(device)
    finally:
        torch.load = _original_torch_load
    
    load_time = time.time() - load_start
    
    # Log model parameters and configuration
    try:
        param_count = sum(p.numel() for p in _plate_model.model.parameters())
        param_count_m = param_count / 1_000_000
        
        logger.info(f"  ✓ YOLO26 plate detector loaded successfully")
        logger.info(f"    - Model parameters: {param_count_m:.2f}M")
        logger.info(f"    - Architecture: {'End-to-End (NMS-Free)' if not has_nms else 'Traditional'}")
        logger.info(f"    - Load time: {load_time:.2f}s")
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
    
    # Run a quick inference test to measure performance
    
    return _plate_model


def _get_plate_model():
    """Return the already-loaded YOLO model (singleton)."""
    global _plate_model
    if _plate_model is None:
        _plate_model = load_plate_model()
    
    return _plate_model


# ---- PaddleOCR PP-OCRv5 ----

def load_paddleocr_model():
    """Load the PaddleOCR PP-OCRv5 model once at startup.

    If PaddleOCR is not installed or fails to load, logs a warning ONCE and
    sets the instance to sentinel _PADDLE_UNAVAILABLE (graceful degradation).
    Thread-safe via double-checked locking.
    """
    global _paddleocr_instance

    # Fast path: already loaded or already failed
    if _paddleocr_instance is not None:
        return None if _paddleocr_instance is _PADDLE_UNAVAILABLE else _paddleocr_instance

    with _paddle_lock:
        # Re-check after acquiring lock (another thread may have loaded it)
        if _paddleocr_instance is not None:
            return None if _paddleocr_instance is _PADDLE_UNAVAILABLE else _paddleocr_instance

        if not _PADDLEOCR_AVAILABLE or _PaddleOCR is None:
            logger.warning(
                "PaddleOCR is not installed — OCR will fall back to EasyOCR. "
                "Install with: pip install paddleocr"
            )
            _paddleocr_instance = _PADDLE_UNAVAILABLE
        else:
            try:
                logger.info("Loading PaddleOCR PP-OCRv5 model...")
                load_start = time.time()

                # Try newer API first (show_log removed in newer versions)
                try:
                    _paddleocr_instance = _PaddleOCR(
                        use_angle_cls=True,
                        lang="en",
                    )
                except TypeError:
                    _paddleocr_instance = _PaddleOCR(
                        use_angle_cls=True,
                        lang="en",
                        show_log=False,
                    )

                load_time = time.time() - load_start
                logger.info(f"  ✓ PaddleOCR PP-OCRv5 loaded successfully (load time: {load_time:.2f}s)")
            except Exception as e:
                import sys
                py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                if "No module named 'paddle'" in str(e):
                    logger.warning(
                        f"PaddleOCR (paddlepaddle) is not installed. To fix this, run: pip install paddlepaddle paddleocr. "
                        f"NOTE: PaddlePaddle currently supports Python 3.9-3.13. Your version ({py_version}) may be too new."
                    )
                else:
                    logger.warning(f"PaddleOCR failed to load: {e} — falling back to EasyOCR")
                _paddleocr_instance = _PADDLE_UNAVAILABLE

    return None if _paddleocr_instance is _PADDLE_UNAVAILABLE else _paddleocr_instance


def recognize_plate_paddleocr(plate_image: np.ndarray) -> Tuple[str, float]:
    """Run PaddleOCR PP-OCRv5 inference on a plate crop.

    Args:
        plate_image: Plate crop as a numpy array (BGR or grayscale).

    Returns:
        (text, confidence) tuple. Returns ("", 0.0) when PaddleOCR is
        unavailable, returns no results, or confidence is below the
        minimum threshold (config.OCR_CONFIDENCE_THRESHOLD = 0.25).
    """
    global _paddleocr_instance

    if _paddleocr_instance is None or _paddleocr_instance is _PADDLE_UNAVAILABLE:
        _paddleocr_instance = load_paddleocr_model() or _PADDLE_UNAVAILABLE

    if _paddleocr_instance is _PADDLE_UNAVAILABLE:
        logger.debug("PaddleOCR unavailable — skipping inference")
        return ("", 0.0)

    if plate_image is None or plate_image.size == 0:
        logger.debug("Empty plate image passed to recognize_plate_paddleocr")
        return ("", 0.0)

    inference_start = time.time()
    try:
        # In PaddleOCR 3.x, passing cls=True to .ocr() can cause 'unexpected keyword argument' 
        # because the classifier is already enabled/managed by the constructor's 'use_angle_cls' flag.
        results = _paddleocr_instance.ocr(plate_image)
    except Exception as e:
        logger.error(f"PaddleOCR inference error: {e}")
        return ("", 0.0)

    inference_time = (time.time() - inference_start) * 1000  # ms

    # PaddleOCR returns: [[[box, (text, confidence)], ...]] or None
    if not results or results[0] is None:
        logger.debug(f"PaddleOCR returned no results (inference: {inference_time:.1f}ms)")
        return ("", 0.0)

    # Collect all text lines and calculate average confidence
    texts = []
    confs = []

    for line in results[0]:
        if not line or len(line) < 2:
            continue
        text_conf = line[1]
        if not text_conf or len(text_conf) < 2:
            continue
        texts.append(str(text_conf[0]))
        confs.append(float(text_conf[1]))

    if not texts:
        return ("", 0.0)
        
    best_text = " ".join(texts)
    best_conf = sum(confs) / len(confs)

    # Apply minimum confidence threshold
    if best_conf < config.OCR_CONFIDENCE_THRESHOLD:
        logger.debug(
            f"PaddleOCR confidence {best_conf:.3f} below threshold "
            f"{config.OCR_CONFIDENCE_THRESHOLD} — rejecting '{best_text}' "
            f"(inference: {inference_time:.1f}ms)"
        )
        return ("", 0.0)

    logger.info(
        f"PaddleOCR result: '{best_text}' (conf={best_conf:.3f}, "
        f"inference={inference_time:.1f}ms)"
    )
    return (best_text, best_conf)


# ---- EasyOCR Engine ----

def load_easyocr_model():
    """Load EasyOCR reader once at startup (singleton). Returns instance or None."""
    global _easyocr_instance
    if _easyocr_instance is not None:
        return _easyocr_instance
    try:
        import easyocr  # type: ignore
        logger.info("Loading EasyOCR model...")
        t = time.time()
        _easyocr_instance = easyocr.Reader(['en'], gpu=False)
        # Warm up with a dummy image so first real call is fast
        dummy = np.zeros((40, 120, 3), dtype=np.uint8)
        _easyocr_instance.readtext(dummy)
        logger.info(f"  ✓ EasyOCR loaded and warmed up ({time.time()-t:.1f}s)")
    except ImportError:
        logger.debug("EasyOCR not installed")
        _easyocr_instance = None
    except Exception as e:
        logger.warning(f"EasyOCR failed to load: {e}")
        _easyocr_instance = None
    return _easyocr_instance


def recognize_plate_easyocr(plate_image: np.ndarray) -> Tuple[str, float]:
    """Run EasyOCR inference on a plate crop.

    Args:
        plate_image: Plate crop as a numpy array (BGR or grayscale).

    Returns:
        (text, confidence) tuple. Returns ("", 0.0) when EasyOCR is
        unavailable, returns no results, or confidence is below threshold.
    """
    global _easyocr_instance

    if plate_image is None or plate_image.size == 0:
        logger.debug("Empty plate image passed to recognize_plate_easyocr")
        return ("", 0.0)

    try:
        import easyocr  # type: ignore
    except ImportError:
        logger.debug("EasyOCR is not installed — skipping")
        return ("", 0.0)

    try:
        if _easyocr_instance is None:
            load_easyocr_model()
        if _easyocr_instance is None:
            return ("", 0.0)
    except Exception as e:
        logger.warning(f"EasyOCR failed to initialize: {e}")
        return ("", 0.0)

    try:
        results = _easyocr_instance.readtext(plate_image)
    except Exception as e:
        logger.warning(f"EasyOCR inference error: {e}")
        return ("", 0.0)

    if not results:
        return ("", 0.0)

    # Combine all detected text blocks
    texts = []
    confs = []
    for (_bbox, text, conf) in results:
        texts.append(str(text))
        confs.append(float(conf))
        
    if not texts:
        return ("", 0.0)

    best_text = " ".join(texts)
    best_conf = sum(confs) / len(confs)

    if best_conf < config.OCR_CONFIDENCE_THRESHOLD:
        logger.debug(
            f"EasyOCR confidence {best_conf:.3f} below threshold "
            f"{config.OCR_CONFIDENCE_THRESHOLD} — rejecting '{best_text}'"
        )
        return ("", 0.0)

    logger.info(f"EasyOCR result: '{best_text}' (conf={best_conf:.3f})")
    return (best_text, best_conf)


# ---- Tesseract Engine ----

def recognize_plate_tesseract(plate_image: np.ndarray) -> Tuple[str, float]:
    """Run Tesseract OCR inference on a plate crop.

    Args:
        plate_image: Plate crop as a numpy array (BGR or grayscale).

    Returns:
        (text, confidence) tuple. Returns ("", 0.0) when Tesseract is
        unavailable, returns no results, or confidence is below threshold.
    """
    global _tesseract_available

    if plate_image is None or plate_image.size == 0:
        logger.debug("Empty plate image passed to recognize_plate_tesseract")
        return ("", 0.0)

    if _tesseract_available is False:
        return ("", 0.0)

    try:
        import pytesseract  # type: ignore
        _tesseract_available = True
    except ImportError:
        logger.debug("pytesseract is not installed — skipping")
        _tesseract_available = False
        return ("", 0.0)

    try:
        tess_config = (
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        data = pytesseract.image_to_data(
            plate_image,
            output_type=pytesseract.Output.DICT,
            config=tess_config,
        )
    except Exception as e:
        logger.warning(f"Tesseract inference error: {e}")
        return ("", 0.0)

    # Collect words with valid confidence
    texts_found = []
    confs_found = []
    texts = data.get('text', [])
    confs = data.get('conf', [])

    for raw_text, raw_conf in zip(texts, confs):
        try:
            conf_val = float(raw_conf)
        except (ValueError, TypeError):
            continue
        if conf_val < 0:
            continue
        # Tesseract confidence is 0-100; normalize to 0.0-1.0
        conf_norm = conf_val / 100.0
        word = str(raw_text).strip()
        if word:
            texts_found.append(word)
            confs_found.append(conf_norm)
            
    if not texts_found:
        return ("", 0.0)

    best_text = " ".join(texts_found)
    best_conf = sum(confs_found) / len(confs_found)

    if best_conf < config.OCR_CONFIDENCE_THRESHOLD:
        logger.debug(
            f"Tesseract confidence {best_conf:.3f} below threshold "
            f"{config.OCR_CONFIDENCE_THRESHOLD} — rejecting '{best_text}'"
        )
        return ("", 0.0)

    logger.info(f"Tesseract result: '{best_text}' (conf={best_conf:.3f})")
    return (best_text, best_conf)


# ---- OCR Fallback Chain ----

def _run_ocr_with_fallback(plate_image: np.ndarray) -> Tuple[str, float]:
    """Arbitrates OCR engines using Bayesian Posterior probabilities.
    
    Runs PaddleOCR primarily. If its posterior confidence falls below the
    threshold, it invokes EasyOCR and performs a joint probability update 
    or overriding selection based on Bayesian weights.

    Args:
        plate_image: Plate crop as a numpy array.

    Returns:
        (text, confidence) from the Bayesian Arbitration.
    """
    try:
        text, conf = _arbitrator.arbitrate(
            plate_image,
            plate_image, # raw_crop isn't explicitly separated deeper down in this abstraction, so pass same image
            recognize_plate_paddleocr,
            recognize_plate_easyocr
        )
        return text, conf
    except Exception as e:
        logger.error(f"Bayesian Arbitration failed: {e}")
        return ("", 0.0)


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

    # Check minimum dimensions
    if w < config.MIN_PLATE_WIDTH or h < config.MIN_PLATE_HEIGHT:
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
    Generate preprocessing variants of a plate crop for ensemble OCR.
    
    Returns:
        List of 4 variants: [Original, CLAHE, Sharpened, Thresholded]
    """
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()

    # Base upscale for all variants — preserve aspect ratio for 2-line plates
    h, w = gray.shape
    if w < 320:
        scale = 320 / w
        gray = cv2.resize(gray, (320, max(80, int(h * scale))), interpolation=cv2.INTER_CUBIC)
    
    variants = []
    
    # Variant 1: Original Grayscale (Contrast normalized)
    variants.append(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX))
    
    # Variant 2: CLAHE enhanced (Req 7.2)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(enhanced)
    
    # Variant 3: Sharpened + CLAHE (Req 7.3)
    kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    variants.append(sharpened)
    
    # Variant 4: Adaptive threshold (binary) (Req 7.4)
    thresh = _apply_multiple_thresholds(sharpened)
    variants.append(thresh)

    # Variant 5: Glare Reduced (Gamma Correction) - Better for mobile screens
    invGamma = 1.0 / 0.7
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    glare_reduced = cv2.LUT(enhanced, table)
    variants.append(glare_reduced)

    # Variant 6: Negative (Inverted) - Sometimes much better for dark plates
    inverted = cv2.bitwise_not(enhanced)
    variants.append(inverted)

    # Variant 7: Deskewed (Auto-rotate)
    try:
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            (h_s, w_s) = gray.shape[:2]
            center = (w_s // 2, h_s // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(enhanced, M, (w_s, h_s), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            variants.append(rotated)
    except:
        pass
    
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

    Pipeline (Requirements 7.1–7.6):
      1. Bilateral filtering  — noise reduction while preserving edges (Req 7.1)
      2. CLAHE                — shadow / glare handling (Req 7.2)
      3. Sharpening kernel    — character edge enhancement (Req 7.3)
      4. Adaptive thresholding (Gaussian, Mean, OTSU best-pick) (Req 7.4, 7.5)
      5. Resize to 320×120 with cubic interpolation (Req 7.6)
    """
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()

    # Step 1: Bilateral filtering — smooths noise while keeping character edges sharp
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Step 2: CLAHE — adaptive histogram equalization for shadow/glare handling
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 3: Sharpening kernel — enhance character edges
    kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

    # Step 4: Adaptive thresholding — Gaussian, Mean, OTSU; best by edge density
    thresh = _apply_multiple_thresholds(sharpened)

    # Morphological closing to connect broken character strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 5: Resize preserving aspect ratio — CRITICAL for 2-line square plates.
    # Forcing 320x120 squishes 2-line plates and destroys character proportions.
    h_crop, w_crop = thresh.shape[:2]
    target_w = 320
    target_h = max(80, int(target_w * h_crop / w_crop)) if w_crop > 0 else 120
    return cv2.resize(thresh, (target_w, target_h), interpolation=cv2.INTER_CUBIC)



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
    
    # ---- Kalman Tracking Logic ----
    # Check if we can skip YOLO inference using existing tracks
    active_tracks = []
    should_skip_yolo = False
    
    # If we have reliable tracks that are NOT due for a YOLO re-sync
    current_tracks = _tracker_manager.trackers
    if current_tracks:
        # Check if ALL current trackers are within the "skip" window
        can_skip_all = all(t.frames_since_update < config.KALMAN_SKIP_FRAMES for t in current_tracks.values())
        if can_skip_all and len(current_tracks) > 0:
            should_skip_yolo = True
            logger.debug(f"Frame {frame_number}: Skipping YOLO inference, using Kalman prediction.")

    if should_skip_yolo:
        # PURE PREDICTION MODE
        results = [] # No YOLO results to process
        detections = [] # Initialize detections array to prevent UnboundLocalError
        # We will populate detections directly from Kalman predictions below
        for t_id, tracker in current_tracks.items():
            pred_bbox = tracker.predict() # x, y, w, h
            x, y, w, h = map(int, pred_bbox)
            
            # Extract crop from prediction
            h_img, w_img = frame.shape[:2]
            clamped_x1 = max(0, x)
            clamped_y1 = max(0, y)
            clamped_x2 = min(w_img, x + w)
            clamped_y2 = min(h_img, y + h)
            
            if (clamped_x2 - clamped_x1) > 10 and (clamped_y2 - clamped_y1) > 10:
                raw_crop = frame[clamped_y1:clamped_y2, clamped_x1:clamped_x2]
                if raw_crop.size > 0:
                    processed = preprocess_plate_crop(raw_crop)
                    detections.append({
                        'bbox': [clamped_x1, clamped_y1, clamped_x2 - clamped_x1, clamped_y2 - clamped_y1],
                        'crop': processed,
                        'raw_crop': raw_crop,
                        'confidence': 0.95, # High confidence for tracked prediction
                        'track_id': t_id,
                        'source': 'kalman_prediction'
                    })
        return detections

    # ---- YOLO INFERENCE MODE ----
    # YOLOv26 Specific Optimization: Use torch.inference_mode context
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.inference_mode():
        model_results = model(
            frame,
            device=device,
            verbose=False,
            conf=config.YOLO_CONFIDENCE_THRESHOLD,
            imgsz=config.YOLO_IMAGE_SIZE,
            half=_use_half_precision,
            # YOLOv26 End-to-End models handle NMS internally
        )
    
    # Process YOLO results into temporal trackers
    raw_bboxes = []
    yolo_detections_temp = []
    detections: List[Dict[str, Any]] = []
    filtered_count: int = 0
    detection_idx: int = 0

    # Get class name mapping from model
    class_names: Dict[int, str] = getattr(model, 'names', {}) or {}


    for result in model_results:
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
            raw_bboxes.append([clamped_x1, clamped_y1, w_box, h_box])
            detection_idx += 1 # type: ignore

    # Update Kalman trackers with new YOLO results
    tracked_results = _tracker_manager.update(raw_bboxes)
    
    # Associate track IDs with our final detection list
    for det in detections:
        # Match back to track ID based on centroid/bbox similarity
        best_id = -1
        min_dist = 50
        det_ctr = (det['bbox'][0] + det['bbox'][2]/2, det['bbox'][1] + det['bbox'][3]/2)
        
        for track_info in tracked_results:
            t_id = track_info[0]
            t_bbox = track_info[1]
            t_ctr = (t_bbox[0] + t_bbox[2]/2, t_bbox[1] + t_bbox[3]/2)
            dist = ((det_ctr[0]-t_ctr[0])**2 + (det_ctr[1]-t_ctr[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                best_id = t_id
        
        det['track_id'] = best_id if best_id != -1 else f"temp_{int(time.time()*1000)}"
        det['source'] = 'yolo_detection'

    if filtered_count > 0:
        logger.debug(f"Filtered {filtered_count} plates by geometric constraints")

    logger.info(f"Frame {frame_number}: Detected {len(detections)} plates (YOLO Refreshed)")
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

    # Automatically correct 0/O, 8/B based on structural Indian plate positions
    text = _apply_position_based_corrections(text)

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
        # Indian plates have max 4 digits for registration number.
        # Everything before that is series.
        suffix_len = len(suffix)
        min_series_len = max(1, suffix_len - 4)
        
        series_end = min_series_len
        for j in range(min_series_len, suffix_len):
            ch_suffix = str(suffix[j])
            if ch_suffix.isdigit():
                # Found first digit after the guaranteed series bounds
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
                    # If position must be letter, map '0' to 'D' in series as 'D' is standard for Indian plates, or 'O' if unavailable.
                    digit_to_letter = {'0': 'D', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G', '4': 'A', '7': 'T'}
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
    
    result = ''.join(corrected)
    
    # ---- State Code Snap-Correction ----
    # Validate state code against 36 known RTO codes.
    # If OCR read a near-miss (e.g. TA instead of TN), snap to the closest valid code.
    VALID_STATE_CODES = {
        'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN',
        'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD',
        'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ',
        'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
    }
    
    state_code = result[:2]
    if state_code not in VALID_STATE_CODES and len(result) >= 6:
        def _state_dist(a: str, b: str) -> int:
            return sum(1 for x, y in zip(a, b) if x != y)
        
        best_code = None
        best_dist = 99
        for code in VALID_STATE_CODES:
            d = _state_dist(state_code, code)
            if d < best_dist:
                best_dist = d
                best_code = code
        
        if best_code and best_dist <= 1:  # Only snap on single-char misread
            logger.debug(f"State code snap: '{state_code}' → '{best_code}' (dist={best_dist})")
            result = best_code + result[2:]
    
    return result



def read_plate(plate_image: np.ndarray, preprocessed_image: np.ndarray = None) -> Tuple[Optional[str], float]:
    """
    Read text from a plate crop using fast parallel ensemble OCR.

    Runs 2 preprocessing variants concurrently (Original + CLAHE) and
    returns as soon as any variant hits confidence >= 0.7.
    Falls back to best result.
    """
    if plate_image is None or plate_image.size == 0:
        return None, 0.0

    # Use all 4 variants for maximum accuracy (Requirements 3.5, 7.7)
    variants = preprocess_plate_variants(plate_image)
    variant_names = ["original", "CLAHE", "sharpened", "thresholded"]
    FAST_EXIT_CONF = 0.82  # Return immediately only on very strong OCR

    def _ocr_variant(idx_variant):
        idx, variant = idx_variant
        name = variant_names[idx] if idx < len(variant_names) else f"variant_{idx}"
        try:
            raw_text, confidence = _run_ocr_with_fallback(variant)
            if not raw_text or confidence < config.OCR_CONFIDENCE_THRESHOLD:
                return None

            # Basic cleaning for ensemble comparison
            cleaned = clean_text(raw_text)
            if is_garbage_text(cleaned):
                return None

            logger.debug(f"OCR variant result: '{raw_text}' -> cleaned: '{cleaned}' (conf: {confidence:.3f})")

            # Score variants based on confidence and "platiness" (matches Indian pattern)
            score = float(confidence)
            # Indian RTO pattern: SS NN SS NNNN (with some flexibility)
            if re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]{2,8}$', cleaned):
                score *= 1.35 # 35% boost for matching the expected structure
            
            return {
                "text": cleaned,
                "raw_text": raw_text,
                "confidence": float(confidence),
                "variant": name,
                "score": score,
                "index": idx
            }
        except Exception as e:
            logger.error(f"OCR variant {idx} error: {e}")
            return None

    candidates = []
    start_ensemble = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_ocr_variant, (i, v)): i for i, v in enumerate(variants)}
        for future in as_completed(futures):
            result = future.result()
            if result:
                candidates.append(result)
                # Fast exit only for high-confidence candidates that are
                # not pattern mismatches; this improves correctness.
                if (
                    float(result.get("confidence", 0.0)) >= FAST_EXIT_CONF
                    and re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]{2,8}$', result.get("text", ""))
                ):
                    logger.info(f"Ensemble FAST-EXIT on variant '{result['variant']}' (conf={result['confidence']:.3f})")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    ensemble_time = time.time() - start_ensemble
    logger.info(f"Ensemble OCR completed in {ensemble_time:.3f}s with {len(candidates)} candidates")

    if not candidates:
        logger.debug("Ensemble OCR: all variants returned no usable result")
        return None, 0.0

    # Sort by score (desc), then by confidence (desc), then by index (asc) for determinism
    candidates.sort(key=lambda x: (-x.get("score", 0), -x.get("confidence", 0), x.get("index", 0)))
    best = candidates[0]
    logger.info(
        f"Ensemble OCR selected variant '{best['variant']}': "
        f"'{best['text']}' (conf={best['confidence']:.3f}, score={best['score']:.3f}) "
        f"from {len(candidates)} candidate(s)"
    )
    return cast(str, best["text"]), float(best["confidence"])


def get_read_confidence(plate_image: np.ndarray, preprocessed_image: np.ndarray = None) -> float:
    """
    Get the OCR recognition confidence for a plate image.
    
    NOTE: This now delegates to read_plate() to avoid running OCR twice.
    The caller should prefer using the confidence returned by read_plate() directly.
    """
    if plate_image is None or plate_image.size == 0:
        return 0.0

    try:
        _, confidence = read_plate(plate_image, preprocessed_image)
        return float(confidence) if confidence is not None else 0.0
    except Exception as e:
        logger.error(f"OCR confidence error: {e}")
        return 0.0
