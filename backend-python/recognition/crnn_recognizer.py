"""
RoadVision — License Plate OCR Recognizer
Optimized multi-engine OCR with intelligent fallback for Indian plates.

OCR Engine Priority (for Indian plates):
  1. PaddleOCR (threshold: 0.5) - Best accuracy for Indian plates
  2. EasyOCR (threshold: 0.4) - Good secondary deep learning option
  3. Tesseract (threshold: 0.3) - Final fallback for simple text
  4. CRNN (if available) - Custom training fallback

Character set: 0-9 A-Z (36 characters + CTC blank).
Input: preprocessed grayscale plate crop (resized to 100×32 for CRNN).
Output: (text, confidence) tuple.
"""

import logging
import os
from typing import Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---- Configuration ----
IMG_HEIGHT = 32
IMG_WIDTH = 100
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHARACTERS) + 1  # +1 for CTC blank token
BLANK_IDX = 0  # CTC blank is index 0

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'crnn.pth'
)

# Lazy-loaded model
_crnn_model = None
_torch_available = False

try:
    import torch
    import torch.nn as nn
    _torch_available = True
except ImportError:
    logger.warning("PyTorch not installed — CRNN recognizer will be unavailable.")


# ---- CRNN Architecture ----
if _torch_available:
    class CRNN(nn.Module):
        """
        Convolutional Recurrent Neural Network for text recognition.

        Architecture:
          Conv Block 1: Conv2d(1,64) → BN → ReLU → MaxPool(2,2)
          Conv Block 2: Conv2d(64,128) → BN → ReLU → MaxPool(2,2)
          Conv Block 3: Conv2d(128,256) → BN → ReLU
          Conv Block 4: Conv2d(256,256) → BN → ReLU → MaxPool(2,1)
          Conv Block 5: Conv2d(256,512) → BN → ReLU
          Conv Block 6: Conv2d(512,512) → BN → ReLU → MaxPool(2,1)
          Conv Block 7: Conv2d(512,512) → BN → ReLU

          BiLSTM: 2 layers, hidden=256

          Linear: 512 → NUM_CLASSES
        """

        def __init__(self, num_classes: int = NUM_CLASSES):
            super().__init__()

            # CNN feature extractor
            self.cnn = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                # Block 2
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                # Block 3
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                # Block 4
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1), (2, 1)),

                # Block 5
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),

                # Block 6
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1), (2, 1)),

                # Block 7
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

            # Bidirectional LSTM
            self.rnn = nn.LSTM(
                input_size=512,
                hidden_size=256,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )

            # Output projection
            self.fc = nn.Linear(512, num_classes)  # 256*2 (bidirectional)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (batch, 1, H, W) grayscale image tensor

            Returns:
                (batch, seq_len, num_classes) log-probabilities
            """
            # CNN features: (B, 512, 1, W')
            conv = self.cnn(x)

            # Squeeze height dimension and permute to (B, W', 512)
            b, c, h, w = conv.size()
            conv = conv.squeeze(2)  # (B, 512, W')
            conv = conv.permute(0, 2, 1)  # (B, W', 512)

            # BiLSTM
            rnn_out, _ = self.rnn(conv)  # (B, W', 512)

            # Projection
            output = self.fc(rnn_out)  # (B, W', num_classes)

            return output


def _load_model():
    """Load the CRNN model (lazy, singleton)."""
    global _crnn_model

    if _crnn_model is not None:
        return _crnn_model

    if not _torch_available:
        logger.error("PyTorch is not installed. Cannot load CRNN model.")
        return None

    model = CRNN(NUM_CLASSES)

    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading CRNN weights from: {MODEL_PATH}")
        try:
            # Handle PyTorch 2.6+ weights_only behavior safely
            import inspect
            sig = inspect.signature(torch.load)
            if 'weights_only' in sig.parameters:
                state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
            else:
                state = torch.load(MODEL_PATH, map_location='cpu')
            model.load_state_dict(state)
        except Exception as e:
            logger.error(f"Failed to load CRNN weights: {e}")
            try:
                # Fallback to weights_only=False if True fails (for older models on newer PyTorch)
                state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
                model.load_state_dict(state)
                logger.info("Successfully loaded CRNN weights with weights_only=False")
            except:
                return None
    else:
        # Do NOT generate random placeholder weights — CRNN must not run
        # with untrained weights as it produces garbage output.
        logger.warning(
            f"CRNN weights not found at {MODEL_PATH}. "
            "CRNN recognition is DISABLED. "
            "Place a trained crnn.pth file in backend-python/models/ "
            "to enable plate text recognition."
        )
        return None

    model.eval()
    _crnn_model = model
    return _crnn_model


def _ctc_decode(output: "torch.Tensor") -> Tuple[str, float]:
    """
    Greedy CTC decoding: take argmax at each timestep, collapse repeats,
    remove blanks.

    Args:
        output: (seq_len, num_classes) log-probabilities for a single sample.

    Returns:
        (decoded_text, average_confidence)
    """
    # Softmax to get probabilities
    probs = torch.softmax(output, dim=-1)
    max_probs, indices = probs.max(dim=-1)

    chars = []
    confidences = []
    prev_idx = -1

    for t in range(indices.size(0)):
        idx = indices[t].item()
        prob = max_probs[t].item()

        # Skip blanks and repeated characters
        if idx != BLANK_IDX and idx != prev_idx:
            if 1 <= idx <= len(CHARACTERS):
                chars.append(CHARACTERS[idx - 1])  # idx 1-36 → char
                confidences.append(prob)

        prev_idx = idx

    text = ''.join(chars)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return text, avg_conf


def preprocess_for_crnn(plate_image: np.ndarray) -> Optional["torch.Tensor"]:
    """
    Preprocess a plate crop image for CRNN input.

    Steps:
      1. Convert to grayscale (if needed)
      2. Resize to IMG_WIDTH × IMG_HEIGHT
      3. Normalize to [0, 1]
      4. Convert to tensor (1, 1, H, W)

    Args:
        plate_image: BGR or grayscale numpy array.

    Returns:
        Torch tensor (1, 1, 32, 100) or None on failure.
    """
    if not _torch_available:
        return None

    import cv2

    if plate_image is None or plate_image.size == 0:
        return None

    try:
        # Ensure grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        # Resize to fixed CRNN input size
        resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Convert to tensor: (1, 1, H, W)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor

    except Exception as e:
        logger.error(f"CRNN preprocessing error: {e}")
        return None


def recognize_plate(plate_image: np.ndarray, preprocessed_image: np.ndarray = None) -> Tuple[str, float]:
    """
    Recognize text from a plate crop image using the optimal fallback chain.

    Optimized Fallback Chain (EasyOCR Primary):
      1. EasyOCR (threshold: 0.25) - Primary engine for license plates
      2. PaddleOCR (threshold: 0.30) - Secondary deep learning option
      3. Tesseract (threshold: 0.25) - Final fallback for simple text
      4. CRNN (if weights available) - Custom training fallback

    Engine-specific confidence thresholds ensure optimal accuracy while
    maintaining intelligent fallback behavior.

    Args:
        plate_image: BGR plate crop (best for DL models like Paddle/EasyOCR).
        preprocessed_image: Preprocessed grayscale binary image (best for Tesseract).

    Returns:
        (recognized_text, confidence) tuple.
        Returns ("", 0.0) if recognition fails.
    """
    if plate_image is None or plate_image.size == 0:
        return "", 0.0

    # 1. Try EasyOCR first - Best for license plates (threshold: 0.25)
    text, confidence = _easyocr_recognize(plate_image)
    if text and confidence >= 0.25:
        logger.info(f"✓ Final result from EasyOCR: '{text}' (conf: {confidence:.2f})")
        return text, confidence

    # 2. Try PaddleOCR (threshold: 0.30)
    text, confidence = _paddleocr_recognize(plate_image)
    if text and confidence >= 0.30:
        logger.info(f"✓ Final result from PaddleOCR: '{text}' (conf: {confidence:.2f})")
        return text, confidence

    # 3. Try Tesseract (threshold: 0.25)
    # Use preprocessed (binary) image for Tesseract if provided
    tess_input = preprocessed_image if preprocessed_image is not None else plate_image
    text, confidence = _tesseract_recognize(tess_input)
    if text and confidence >= 0.25:
        logger.info(f"✓ Final result from Tesseract: '{text}' (conf: {confidence:.2f})")
        return text, confidence

    # 4. Try Custom CRNN as final fallback (if weights available)
    model = _load_model()
    if model is not None:
        tensor = preprocess_for_crnn(plate_image)
        if tensor is not None:
            try:
                with torch.no_grad():
                    output = model(tensor)  # (1, seq_len, num_classes)
                    text, confidence = _ctc_decode(output[0])
                    if text:
                        logger.info(f"✓ Final result from CRNN: '{text}' (conf: {confidence:.2f})")
                        return text, confidence
            except Exception as e:
                logger.error(f"CRNN recognition error: {e}")

    # All engines failed
    logger.warning("All OCR engines failed to recognize plate text")
    return "", 0.0


# ---- PaddleOCR Fallback (replaces EasyOCR for better Indian plate accuracy) ----
_paddle_reader = None


def _get_paddle_reader():
    """Lazy initialize PaddleOCR reader (heavy initialization).

    Uses angle classification (use_angle_cls=True) to handle tilted
    plates common on Indian two-wheelers and auto-rickshaws.
    """
    global _paddle_reader
    if _paddle_reader is None:
        try:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR reader (CRNN fallback)...")
            # Use minimal parameters for compatibility
            _paddle_reader = PaddleOCR(
                lang='en',
                use_angle_cls=True,
            )
            logger.info("PaddleOCR reader ready.")
        except ImportError:
            logger.error(
                "Neither CRNN weights nor PaddleOCR are available. "
                "Install paddleocr: pip install paddlepaddle paddleocr"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}", exc_info=True)
            return None
    return _paddle_reader


# ---- EasyOCR Fallback (better than Tesseract for license plates) ----
_easyocr_reader = None


def _get_easyocr_reader():
    """Lazy initialize EasyOCR reader (heavy initialization)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            logger.info("Initializing EasyOCR reader...")
            _easyocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR reader ready.")
        except ImportError:
            logger.error("EasyOCR not available. Install: pip install easyocr")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            return None
    return _easyocr_reader


def _easyocr_recognize(plate_image: np.ndarray) -> Tuple[str, float]:
    """
    Recognize plate text using EasyOCR (best for license plates).
    
    Returns:
        (text, confidence) tuple.
    """
    reader = _get_easyocr_reader()
    if reader is None:
        return "", 0.0

    try:
        logger.debug("Attempting EasyOCR recognition...")
        # Enhanced preprocessing specifically for EasyOCR
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Aggressive upscaling for better character recognition
        h, w = gray.shape
        target_height = 120
        if h < target_height:
            scale = target_height / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            h, w = gray.shape
        
        # Advanced denoising
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Strong CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Sharpening to enhance edges
        kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # Convert back to BGR for EasyOCR
        bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        # Try with very permissive settings
        results = reader.readtext(
            bgr,
            detail=1,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            batch_size=1,
            width_ths=0.5,  # More permissive word grouping
            height_ths=0.5,
            decoder='greedy'  # Faster, often better for plates
        )
        
        if not results:
            logger.debug("EasyOCR: No text detected")
            return "", 0.0
        
        # Combine all detected text
        texts = []
        confidences = []
        for (bbox, text, conf) in results:
            if conf >= 0.15:  # Very low threshold to catch more text
                texts.append(text.strip())
                confidences.append(conf)
        
        if not texts:
            logger.debug("EasyOCR: No text with sufficient confidence (threshold: 0.15)")
            return "", 0.0
        
        combined_text = ''.join(texts).upper()
        avg_conf = sum(confidences) / len(confidences)
        
        logger.debug(f"EasyOCR detected: '{combined_text}' (conf: {avg_conf:.2f})")
        return combined_text, avg_conf
        
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}")
        return "", 0.0


def _paddleocr_recognize(plate_image: np.ndarray) -> Tuple[str, float]:
    """
    Recognize plate text using PaddleOCR as a fallback.
    If PaddleOCR fails, falls back to Tesseract.

    Returns:
        (text, confidence) tuple.
    """
    reader = _get_paddle_reader()
    if reader is None:
        return _tesseract_recognize(plate_image)

    try:
        logger.debug("Attempting PaddleOCR recognition...")
        
        # Enhanced preprocessing for PaddleOCR
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Upscale significantly for better OCR
        h, w = gray.shape
        if w < 300 or h < 80:
            scale = max(300 / w, 80 / h)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply strong CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Sharpen
        kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # Convert back to BGR for PaddleOCR
        bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        # PaddleOCR 3.4+ uses PaddleX format - call without cls parameter
        results = reader.ocr(bgr)

        if not results or not results[0]:
            logger.debug("PaddleOCR returned empty results")
            return "", 0.0

        texts = []
        confidences = []
        
        # Handle both PaddleX dict format and standard list format
        if isinstance(results[0], dict):
            # PaddleX 3.4+ format: {'rec_texts': [...], 'rec_scores': [...]}
            rec_texts = results[0].get('rec_texts', [])
            rec_scores = results[0].get('rec_scores', [])
            for text, conf in zip(rec_texts, rec_scores):
                if conf >= 0.3:  # Lowered threshold
                    texts.append(str(text))
                    confidences.append(float(conf))
        elif isinstance(results[0], list):
            # Standard PaddleOCR format: [[[box], (text, conf)], ...]
            for line in results[0]:
                if line and len(line) >= 2:
                    if isinstance(line[1], (tuple, list)) and len(line[1]) >= 2:
                        text, conf = line[1][0], line[1][1]
                        if conf >= 0.3:  # Lowered threshold
                            texts.append(str(text))
                            confidences.append(float(conf))

        if not texts:
            logger.debug("PaddleOCR: No text with sufficient confidence (threshold: 0.3)")
            return "", 0.0

        combined_text = ''.join(texts).strip().upper()
        avg_conf = sum(confidences) / len(confidences)
        logger.debug(f"PaddleOCR detected: '{combined_text}' (conf: {avg_conf:.2f})")
        return combined_text, avg_conf

    except Exception as e:
        logger.warning(f"PaddleOCR failed: {e}")
        return "", 0.0


def _tesseract_recognize(plate_image: np.ndarray) -> Tuple[str, float]:
    """
    Fallback OCR using Tesseract with enhanced preprocessing.
    
    Returns:
        (text, confidence) tuple.
    """
    try:
        import pytesseract
        from PIL import Image
        
        logger.debug("Attempting Tesseract recognition...")
        
        # Robust Tesseract path resolution for Windows
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Users\Sivab\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
            os.path.join(os.environ.get('LOCALAPPDATA', ''), r'Programs\Tesseract-OCR\tesseract.exe'),
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        tess_path = None
        for p in possible_paths:
            if os.path.exists(p):
                tess_path = p
                break
        
        if tess_path:
            pytesseract.pytesseract.tesseract_cmd = tess_path
        else:
            # Check if tesseract is in PATH
            import shutil
            if not shutil.which("tesseract"):
                logger.warning("Tesseract executable not found in common paths or PATH. Tesseract fallback will likely fail.")
        
        # Enhanced preprocessing for Tesseract
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Upscale if too small
        h, w = gray.shape
        if w < 200 or h < 60:
            scale = max(200 / w, 60 / h)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while keeping edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Try adaptive threshold (often better than OTSU for plates)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert to PIL
        pil_img = Image.fromarray(thresh)
        
        # Try multiple PSM modes
        configs = [
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]
        
        best_text = ""
        best_conf = 0.0
        
        for config in configs:
            try:
                data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
                
                texts = []
                confidences = []
                for i, conf in enumerate(data['conf']):
                    if conf > 40 and data['text'][i].strip():  # Lower threshold
                        texts.append(data['text'][i].strip())
                        confidences.append(conf / 100.0)
                
                if texts:
                    combined_text = ''.join(texts).upper()
                    avg_conf = sum(confidences) / len(confidences)
                    
                    # Keep the result with highest confidence
                    if avg_conf > best_conf:
                        best_text = combined_text
                        best_conf = avg_conf
            except:
                continue
        
        if not best_text:
            logger.debug("Tesseract: No text detected")
            return "", 0.0
        
        logger.debug(f"Tesseract detected: '{best_text}' (conf: {best_conf:.2f})")
        return best_text, best_conf
        
    except ImportError:
        logger.error("Tesseract not available. Install: pip install pytesseract pillow")
        return "", 0.0
    except Exception as e:
        logger.error(f"Tesseract recognition error: {e}")
        return "", 0.0

