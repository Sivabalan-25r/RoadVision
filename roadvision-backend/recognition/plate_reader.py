"""
RoadVision — Plate Text Reader (EasyOCR)
Reads characters from cropped license plate images.
Includes post-processing: confidence filtering, text cleaning, garbage rejection.
Will be replaced with CRNN model in a future upgrade.
"""

import logging
import re
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-load EasyOCR reader (heavy initialization)
_reader = None

# ---- Configuration ----
MIN_OCR_CONFIDENCE = 0.5       # Discard results below this
MIN_PLATE_WIDTH = 60           # Minimum crop width in pixels
MIN_PLATE_HEIGHT = 20          # Minimum crop height in pixels
MIN_CLEANED_LENGTH = 6         # Minimum characters after cleaning
MAX_CLEANED_LENGTH = 12        # Maximum characters (Indian plates are 9-10)

# ---- OCR Character Correction ----
# Common EasyOCR misreads for license plate characters.
# Applied BEFORE context-aware normalization in plate_rules.
OCR_CORRECTIONS = {
    '|': '1',   # Pipe misread as 1
    'l': '1',   # Lowercase L misread as 1
    'i': '1',   # Lowercase i misread as 1
    'o': '0',   # Lowercase o misread as 0
    'q': '0',   # Lowercase q misread as 0
    's': '5',   # Lowercase s misread as 5
    'z': '2',   # Lowercase z misread as 2
    'b': '8',   # Lowercase b misread as 8
    'g': '6',   # Lowercase g misread as 6
    '(': 'C',   # Parenthesis misread as C
    ')': 'J',   # Parenthesis misread as J
    '[': 'L',   # Bracket misread as L
    ']': 'J',   # Bracket misread as J
}

# Uppercase ambiguous character map (applied AFTER uppercasing).
# Used by normalize_plate() as a lighter standalone normalizer.
CHAR_NORMALIZE_MAP = {
    'I': '1',
    'O': '0',
    'B': '8',
    'S': '5',
    'Z': '2',
}

# Known false-positive strings that OCR frequently produces from
# non-plate regions (sign text, headers, stickers, etc.)
KNOWN_FALSE_POSITIVES = {
    'IND', 'INDIA', 'TEST', 'SAMPLE', 'DEMO',
    'GOVT', 'POLICE', 'TAXI', 'AUTO',
}


def _get_reader():
    """Lazy initialize EasyOCR reader (downloads model on first run)."""
    global _reader
    if _reader is None:
        import easyocr
        logger.info("Initializing EasyOCR reader (first load may download models)...")
        _reader = easyocr.Reader(
            ['en'],
            gpu=False,  # CPU-only for portability
            verbose=False,
        )
        logger.info("EasyOCR reader ready.")
    return _reader


def clean_text(text: str) -> str:
    """
    Clean raw OCR text for plate processing.

    Steps:
      1. Convert to uppercase
      2. Apply OCR character corrections
      3. Remove spaces, hyphens, dots
      4. Remove non-alphanumeric characters
    """
    text = text.strip().upper()

    # Apply character corrections (for misreads that uppercase won't fix)
    corrected = []
    for ch in text:
        lower = ch.lower()
        if lower in OCR_CORRECTIONS:
            corrected.append(OCR_CORRECTIONS[lower])
        else:
            corrected.append(ch)
    text = ''.join(corrected)

    # Remove whitespace, hyphens, dots
    text = re.sub(r'[\s\-\.\,\;\:\'\"]+', '', text)

    # Remove any remaining non-alphanumeric characters
    text = re.sub(r'[^A-Z0-9]', '', text)

    return text


def is_garbage_text(text: str) -> bool:
    """
    Detect garbage OCR results that are clearly not number plates.

    Returns True if the text should be discarded.
    """
    if len(text) < MIN_CLEANED_LENGTH:
        return True

    if len(text) > MAX_CLEANED_LENGTH:
        return True

    # Known false-positive words from signs, stickers, headers
    if text in KNOWN_FALSE_POSITIVES:
        return True

    # All same character (e.g., "0000", "AAAA", "1111")
    if len(set(text)) <= 1:
        return True

    # All digits — plates always have letters
    if text.isdigit():
        return True

    # All letters — plates always have digits
    if text.isalpha():
        return True

    # Repeating patterns of 2 chars (e.g., "ABABAB", "1A1A1A")
    if len(text) >= 6:
        pair = text[:2]
        if text == pair * (len(text) // 2) + pair[:len(text) % 2]:
            return True

    # Too few unique characters relative to length (low entropy)
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

    # Consecutive identical pairs (e.g., "AABB1122")
    if len(text) >= 8:
        pairs = [text[j:j+2] for j in range(0, len(text) - 1, 2)]
        if all(p[0] == p[1] for p in pairs if len(p) == 2):
            return True

    return False


def normalize_plate(text: str) -> str:
    """
    Normalize plate text by cleaning whitespace, punctuation, and
    non-alphanumeric characters.

    This is a lightweight cleaner for deduplication. Context-aware
    character corrections (I→1, O→0, etc.) are handled by
    plate_rules.smart_normalize() which knows digit vs letter positions.
    """
    text = text.upper().strip()
    text = re.sub(r'[\s\-\.\,]+', '', text)
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


def is_valid_crop_size(plate_image: np.ndarray) -> bool:
    """Check if the plate crop meets minimum dimension requirements."""
    if plate_image is None or plate_image.size == 0:
        return False
    h, w = plate_image.shape[:2]
    return w >= MIN_PLATE_WIDTH and h >= MIN_PLATE_HEIGHT


def read_plate(plate_image: np.ndarray) -> Optional[str]:
    """
    Read text from a cropped license plate image.

    Pipeline:
      1. Validate crop dimensions (min 60×20)
      2. Run EasyOCR with confidence filter (≥0.5)
      3. Clean and normalize text
      4. Reject garbage results

    Args:
        plate_image: BGR numpy array of the cropped plate region.

    Returns:
        Cleaned plate text or None if unreadable/garbage.
    """
    if not is_valid_crop_size(plate_image):
        logger.debug("Plate crop too small, skipping OCR")
        return None

    try:
        reader = _get_reader()
        results = reader.readtext(
            plate_image,
            detail=1,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -',
        )

        if not results:
            return None

        # Filter by confidence and concatenate
        texts = []
        for (bbox, text, conf) in results:
            if conf >= MIN_OCR_CONFIDENCE:
                texts.append(text)

        if not texts:
            logger.debug("All OCR results below confidence threshold")
            return None

        combined = ' '.join(texts)

        # Clean the text
        cleaned = clean_text(combined)

        # Reject garbage
        if is_garbage_text(cleaned):
            logger.debug(f"Rejected garbage OCR result: '{cleaned}'")
            return None

        return cleaned

    except Exception as e:
        logger.error(f"EasyOCR read error: {e}")
        return None


def get_read_confidence(plate_image: np.ndarray) -> float:
    """
    Get the average confidence score from EasyOCR for this plate image.

    Returns:
        Float between 0 and 1. Returns 0.0 for invalid crops.
    """
    if not is_valid_crop_size(plate_image):
        return 0.0

    try:
        reader = _get_reader()
        results = reader.readtext(plate_image, detail=1, paragraph=False)

        if not results:
            return 0.0

        confidences = [conf for (_, _, conf) in results if conf >= MIN_OCR_CONFIDENCE]
        return sum(confidences) / len(confidences) if confidences else 0.0

    except Exception as e:
        logger.error(f"EasyOCR confidence error: {e}")
        return 0.0
