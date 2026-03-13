"""
RoadVision — CRNN License Plate Recognizer
CNN feature extractor → BiLSTM sequence modeler → CTC-decoded output.

Character set: 0-9 A-Z (36 characters + CTC blank).
Input: preprocessed grayscale plate crop (resized to 100×32).
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
        state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
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

    Fallback Chain:
      1. CRNN (if weights available) - Best for speed/custom training
      2. PaddleOCR - Best for general accuracy on Indian plates
      3. EasyOCR - Good secondary deep learning option
      4. Tesseract - Final fallback for simple text

    Args:
        plate_image: BGR plate crop (best for DL models like Paddle/EasyOCR).
        preprocessed_image: Preprocessed grayscale binary image (best for Tesseract).

    Returns:
        (recognized_text, confidence) tuple.
        Returns ("", 0.0) if recognition fails.
    """
    if plate_image is None or plate_image.size == 0:
        return "", 0.0

    # 1. Try Custom CRNN if available
    model = _load_model()
    if model is not None:
        tensor = preprocess_for_crnn(plate_image)
        if tensor is not None:
            try:
                with torch.no_grad():
                    output = model(tensor)  # (1, seq_len, num_classes)
                    text, confidence = _ctc_decode(output[0])
                    if text and confidence >= 0.5:
                        logger.info(f"CRNN SUCCESS: '{text}' (conf: {confidence:.2f})")
                        return text, confidence
            except Exception as e:
                logger.error(f"CRNN recognition error: {e}")

    # 2. Try PaddleOCR (fallback 1) - Strongest for Indian plates
    text, confidence = _paddleocr_recognize(plate_image)
    if text and confidence >= 0.5:
        return text, confidence

    # 3. Try EasyOCR (fallback 2)
    text, confidence = _easyocr_recognize(plate_image)
    if text and confidence >= 0.4:
        return text, confidence

    # 4. Try Tesseract (final fallback)
    # Use preprocessed (binary) image for Tesseract if provided
    tess_input = preprocessed_image if preprocessed_image is not None else plate_image
    return _tesseract_recognize(tess_input)


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
        return _tesseract_recognize(plate_image)

    try:
        # Preprocess specifically for EasyOCR
        # EasyOCR works better with larger, contrast-enhanced images
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Upscale significantly if image is small
        h, w = gray.shape
        if w < 200 or h < 60:
            scale = max(200 / w, 60 / h)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR for EasyOCR
        bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Try with lower confidence threshold and more permissive settings
        results = reader.readtext(
            bgr,
            detail=1,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            batch_size=1
        )
        
        if not results:
            logger.debug("EasyOCR: No text detected, trying Tesseract")
            return _tesseract_recognize(plate_image)
        
        # Combine all detected text
        texts = []
        confidences = []
        for (bbox, text, conf) in results:
            if conf >= 0.2:  # Very low threshold to catch more text
                texts.append(text.strip())
                confidences.append(conf)
        
        if not texts:
            logger.debug("EasyOCR: No text with sufficient confidence, trying Tesseract")
            return _tesseract_recognize(plate_image)
        
        combined_text = ''.join(texts).upper()
        avg_conf = sum(confidences) / len(confidences)
        
        logger.info(f"EasyOCR SUCCESS: '{combined_text}' (conf: {avg_conf:.2f})")
        return combined_text, avg_conf
        
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}, falling back to Tesseract")
        return _tesseract_recognize(plate_image)


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
        # Call OCR without cls parameter for compatibility
        results = reader.ocr(plate_image)

        if not results or not results[0]:
            logger.debug("PaddleOCR returned empty results, trying Tesseract")
            return _tesseract_recognize(plate_image)

        # The installed version might return Paddlex dict format or standard list format
        texts = []
        confidences = []
        
        # Handle PaddleX dict format
        if isinstance(results[0], dict) and 'rec_texts' in results[0]:
            rec_texts = results[0].get('rec_texts', [])
            rec_scores = results[0].get('rec_scores', [])
            for text, conf in zip(rec_texts, rec_scores):
                if conf >= 0.5:
                    texts.append(text)
                    confidences.append(conf)
        # Handle standard PaddleOCR format: [[[box], (text, conf)], ...]
        elif isinstance(results[0], list):
            for line in results[0]:
                if line and len(line) >= 2:
                    # line format: [box_coords, (text, confidence)]
                    if isinstance(line[1], tuple) and len(line[1]) >= 2:
                        text, conf = line[1][0], line[1][1]
                        if conf >= 0.5:
                            texts.append(str(text))
                            confidences.append(float(conf))
                    elif isinstance(line[1], list) and len(line[1]) >= 2:
                        text, conf = line[1][0], line[1][1]
                        if conf >= 0.5:
                            texts.append(str(text))
                            confidences.append(float(conf))

        if not texts:
            logger.debug("PaddleOCR: No text with sufficient confidence, trying Tesseract")
            return _tesseract_recognize(plate_image)

        combined_text = ''.join(texts).strip().upper()
        avg_conf = sum(confidences) / len(confidences)
        logger.info(f"PaddleOCR SUCCESS: '{combined_text}' (conf: {avg_conf:.2f})")
        return combined_text, avg_conf

    except Exception as e:
        logger.warning(f"PaddleOCR failed: {e}, falling back to Tesseract")
        return _tesseract_recognize(plate_image)


def _tesseract_recognize(plate_image: np.ndarray) -> Tuple[str, float]:
    """
    Fallback OCR using Tesseract with enhanced preprocessing.
    
    Returns:
        (text, confidence) tuple.
    """
    try:
        import pytesseract
        from PIL import Image
        
        # Robust Tesseract path resolution for Windows
        tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if not os.path.exists(tess_path):
            alt_path = r'C:\Users\Sivab\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
            if os.path.exists(alt_path):
                tess_path = alt_path
        
        if os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path
        else:
            logger.warning(f"Tesseract executable not found at {tess_path}. Tesseract fallback will likely fail.")
        
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
        
        logger.info(f"Tesseract SUCCESS: '{best_text}' (conf: {best_conf:.2f})")
        return best_text, best_conf
        
    except ImportError:
        logger.error("Tesseract not available. Install: pip install pytesseract pillow")
        return "", 0.0
    except Exception as e:
        logger.error(f"Tesseract recognition error: {e}")
        return "", 0.0

