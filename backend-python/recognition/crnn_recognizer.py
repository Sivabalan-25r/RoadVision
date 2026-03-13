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


def recognize_plate(plate_image: np.ndarray) -> Tuple[str, float]:
    """
    Recognize text from a preprocessed plate crop image.

    Uses CRNN if trained weights are available (crnn.pth).
    Falls back to PaddleOCR if CRNN is unavailable.

    Args:
        plate_image: BGR or grayscale numpy array of the plate crop.

    Returns:
        (recognized_text, confidence) tuple.
        Returns ("", 0.0) if recognition fails.
    """
    if plate_image is None or plate_image.size == 0:
        return "", 0.0

    model = _load_model()

    # If CRNN is available, use it
    if model is not None:
        tensor = preprocess_for_crnn(plate_image)
        if tensor is None:
            return "", 0.0

        try:
            with torch.no_grad():
                output = model(tensor)  # (1, seq_len, num_classes)
                text, confidence = _ctc_decode(output[0])
                return text, confidence

        except Exception as e:
            logger.error(f"CRNN recognition error: {e}")
            return "", 0.0

    # Fallback: use PaddleOCR when CRNN weights are not available
    # PaddleOCR needs BGR/RGB images, convert grayscale to BGR if needed
    if len(plate_image.shape) == 2:
        plate_image_bgr = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)
    else:
        plate_image_bgr = plate_image
    
    return _paddleocr_recognize(plate_image_bgr)


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


def _paddleocr_recognize(plate_image: np.ndarray) -> Tuple[str, float]:
    """
    Recognize plate text using PaddleOCR as a fallback.
    PaddleOCR achieves ~90% accuracy on Indian license plates.

    Returns:
        (text, confidence) tuple.
    """
    reader = _get_paddle_reader()
    if reader is None:
        return "", 0.0

    try:
        # Call OCR without cls parameter for compatibility
        results = reader.ocr(plate_image)

        if not results or not results[0]:
            logger.debug("PaddleOCR returned empty results")
            return "", 0.0

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
            logger.debug("PaddleOCR: No text with sufficient confidence")
            return "", 0.0

        combined_text = ''.join(texts).strip().upper()
        avg_conf = sum(confidences) / len(confidences)
        logger.info(f"PaddleOCR SUCCESS: '{combined_text}' (conf: {avg_conf:.2f})")
        return combined_text, avg_conf

    except Exception as e:
        logger.error(f"PaddleOCR recognition error: {e}", exc_info=True)
        return "", 0.0

