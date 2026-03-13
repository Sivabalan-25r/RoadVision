"""
RoadVision — Indian Number Plate Format Rules
Validates detected plate text against Indian RTO formatting standards.
Detects character manipulation, spacing issues, and pattern mismatches.
Includes context-aware normalization that considers character position.
"""

import re
from typing import Optional

# =============================================
# Indian RTO Plate Pattern
# Format: SS NN SS NNNN
#   SS = State code (2 letters)
#   NN = District code (2 digits)
#   SS = Series letters (1-3 letters)
#   NNNN = Registration number (1-4 digits)
# =============================================

# Standard Indian plate regex (allows some variation in series/number length)
PLATE_PATTERN = re.compile(
    r'^([A-Z]{2})'     # State code
    r'(\d{2})'          # District code
    r'([A-Z]{1,3})'    # Series letters
    r'(\d{1,4})$'      # Registration number
)

# Strict Indian plate regex (exact format: AA NN AA NNNN)
STRICT_PLATE_PATTERN = re.compile(
    r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,4}$'
)

# ---- Context-aware character substitution maps ----
# When a character appears in a DIGIT position but is a letter
LETTER_TO_DIGIT = {
    'O': '0',
    'I': '1',
    'L': '1',
    'B': '8',
    'S': '5',
    'Z': '2',
    'G': '6',
    'D': '0',
    'Q': '0',
    'T': '7',
    'A': '4',
}

# When a character appears in a LETTER position but is a digit
DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '8': 'B',
    '5': 'S',
    '2': 'Z',
    '6': 'G',
    '4': 'A',
    '7': 'T',
}

# Valid Indian state codes
VALID_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'GA',
    'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH',
    'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK',
    'TN', 'TR', 'TS', 'UK', 'UP', 'WB',
}


class PlateValidationResult:
    """Result of plate format validation."""

    def __init__(
        self,
        detected_plate: str,
        correct_plate: Optional[str] = None,
        violation: Optional[str] = None,
        confidence_modifier: float = 1.0,
    ):
        self.detected_plate = detected_plate
        self.correct_plate = correct_plate or detected_plate
        self.violation = violation
        self.confidence_modifier = confidence_modifier

    @property
    def is_violation(self) -> bool:
        return self.violation is not None

    def to_dict(self) -> dict:
        return {
            "detected_plate": self.detected_plate,
            "correct_plate": self.correct_plate,
            "violation": self.violation,
        }


def normalize_plate(text: str) -> str:
    """
    Clean plate text: remove spaces, hyphens, dots, commas, and
    non-alphanumeric characters, then convert to uppercase.

    This is a lightweight normalizer used for deduplication and quick
    comparisons. It does NOT apply character substitutions — those are
    handled by smart_normalize() which is position-aware.
    """
    text = re.sub(r'[\s\-\.\,]+', '', text.upper().strip())
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


def smart_normalize(plate: str) -> tuple:
    """
    Context-aware normalization: fix characters based on their expected
    position in the Indian plate format (AA NN AA NNNN).

    Returns:
        (normalized_plate, corrections_made: list)
    """
    if len(plate) < 6:
        return plate, []

    corrected = list(plate)
    corrections = []

    # --- Positions 0-1: MUST be letters (state code) ---
    for i in range(min(2, len(plate))):
        ch = plate[i]
        if ch.isdigit() and ch in DIGIT_TO_LETTER:
            corrected[i] = DIGIT_TO_LETTER[ch]
            corrections.append((i, ch, corrected[i]))

    # --- Positions 2-3: MUST be digits (district code) ---
    for i in range(2, min(4, len(plate))):
        ch = plate[i]
        if ch.isalpha() and ch in LETTER_TO_DIGIT:
            corrected[i] = LETTER_TO_DIGIT[ch]
            corrections.append((i, ch, corrected[i]))

    # --- Position 4+: determine where series ends and number begins ---
    if len(plate) > 4:
        suffix = plate[4:]

        # Find transition: series (letters) → registration number (digits)
        # Walk until we find the first digit
        series_end = 0
        for j, ch in enumerate(suffix):
            if ch.isdigit():
                series_end = j
                break
            # Also consider letters that SHOULD be digits
            if ch.isalpha() and j >= 3:
                # After 3 series letters, remaining should be digits
                series_end = j
                break
        else:
            series_end = min(len(suffix), 3)  # Max 3 series letters

        # Series positions: MUST be letters
        for j in range(series_end):
            idx = 4 + j
            if idx < len(plate):
                ch = plate[idx]
                if ch.isdigit() and ch in DIGIT_TO_LETTER:
                    corrected[idx] = DIGIT_TO_LETTER[ch]
                    corrections.append((idx, ch, corrected[idx]))

        # Registration number positions: MUST be digits
        for j in range(series_end, len(suffix)):
            idx = 4 + j
            if idx < len(plate):
                ch = plate[idx]
                if ch.isalpha() and ch in LETTER_TO_DIGIT:
                    corrected[idx] = LETTER_TO_DIGIT[ch]
                    corrections.append((idx, ch, corrected[idx]))

    return ''.join(corrected), corrections


def check_character_manipulation(plate: str) -> Optional[PlateValidationResult]:
    """
    Check if letters and digits are swapped to evade detection.
    Uses context-aware normalization to determine the expected type at each position.
    """
    if len(plate) < 6:
        return None

    corrected_plate, corrections = smart_normalize(plate)

    if corrections:
        return PlateValidationResult(
            detected_plate=plate,
            correct_plate=corrected_plate,
            violation="Character Manipulation",
            confidence_modifier=0.95,
        )

    return None


def check_spacing_manipulation(original_text: str, normalized: str) -> Optional[PlateValidationResult]:
    """
    Check if the plate has unusual spacing designed to confuse readers.
    Valid plates should have spaces in standard positions or no spaces.
    """
    # If the original text had no spaces, no spacing issue
    if ' ' not in original_text and '-' not in original_text:
        return None

    # Standard Indian spacing patterns (acceptable)
    standard_patterns = [
        re.compile(r'^[A-Z]{2}\s?\d{2}\s?[A-Z]{1,3}\s?\d{1,4}$'),
        re.compile(r'^[A-Z]{2}\s?-?\s?\d{2}\s?-?\s?[A-Z]{1,3}\s?-?\s?\d{1,4}$'),
    ]

    cleaned = original_text.upper().strip()
    for pattern in standard_patterns:
        if pattern.match(cleaned):
            return None

    # Non-standard spacing detected
    m = PLATE_PATTERN.match(normalized)
    if m:
        correct = f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"
    else:
        correct = normalized

    return PlateValidationResult(
        detected_plate=normalized,
        correct_plate=correct,
        violation="Spacing Manipulation",
        confidence_modifier=0.9,
    )


def check_tampered_plate(plate: str) -> Optional[PlateValidationResult]:
    """
    Detect plates where characters appear deliberately altered.
    Signs of tampering:
      - Mix of manipulation types in one plate
      - Characters that are ambiguous across multiple substitutions
      - 2+ corrections needed when the state code is also invalid
    """
    if len(plate) < 8:
        return None

    corrected_plate, corrections = smart_normalize(plate)

    # Lower threshold when the state code is not a valid Indian state,
    # since an invalid state + corrections is a strong tampering signal.
    threshold = 3
    state = corrected_plate[:2] if len(corrected_plate) >= 2 else ''
    if state not in VALID_STATE_CODES:
        threshold = 2

    if len(corrections) >= threshold:
        return PlateValidationResult(
            detected_plate=plate,
            correct_plate=corrected_plate,
            violation="Tampered Plate",
            confidence_modifier=0.75,
        )

    return None


def check_pattern_mismatch(plate: str) -> Optional[PlateValidationResult]:
    """
    Check if the plate matches the standard Indian format.
    Expected: AA NN AA NNNN (state, district, series, number).
    """
    if PLATE_PATTERN.match(plate):
        # Valid format — check if state code is real
        state = plate[:2]
        if state not in VALID_STATE_CODES:
            return PlateValidationResult(
                detected_plate=plate,
                correct_plate=plate,
                violation="Invalid State Code",
                confidence_modifier=0.7,
            )
        return None  # Valid plate

    # Try to fix with smart normalization first
    corrected, corrections = smart_normalize(plate)
    if PLATE_PATTERN.match(corrected) and corrections:
        # Could be fixed — this is a character manipulation, not pattern mismatch
        return None  # Let character manipulation checker handle it

    return PlateValidationResult(
        detected_plate=plate,
        correct_plate=plate,
        violation="Plate Pattern Mismatch",
        confidence_modifier=0.8,
    )


def validate_plate(raw_text: str) -> PlateValidationResult:
    """
    Run all validation rules on a detected plate.

    Detection order (violations checked BEFORE normalization hides them):
      1. Store the original OCR output
      2. Detect spacing manipulation on the raw text (before cleaning)
      3. Apply character normalization (I→1, O→0, B→8, S→5, Z→2)
      4. Compare original vs normalized → Character Manipulation
      5. Validate normalized plate against Indian regex
      6. Return both detected and corrected plates
    """
    # Step 1: Store original
    original_plate = raw_text.strip().upper()

    # Step 2: Detect spacing manipulation BEFORE cleaning
    has_spacing = bool(re.search(r'[\s\-]', original_plate))

    # Step 3: Clean plate text (remove spaces/hyphens/punctuation)
    cleaned = normalize_plate(raw_text)

    if len(cleaned) < 6:
        return PlateValidationResult(
            detected_plate=cleaned,
            violation="Plate Pattern Mismatch",
            confidence_modifier=0.5,
        )

    # Step 3b: Apply context-aware character normalization
    corrected_plate, corrections = smart_normalize(cleaned)

    # Step 4: Determine violations — check in priority order

    # Priority 1: Tampered Plate (3+ character corrections with bad state)
    state = corrected_plate[:2] if len(corrected_plate) >= 2 else ''
    tamper_threshold = 2 if state not in VALID_STATE_CODES else 3
    if len(corrections) >= tamper_threshold:
        return PlateValidationResult(
            detected_plate=cleaned,
            correct_plate=corrected_plate,
            violation="Tampered Plate",
            confidence_modifier=0.75,
        )

    # Priority 2: Character Manipulation (original differs from corrected)
    if corrections:
        return PlateValidationResult(
            detected_plate=cleaned,
            correct_plate=corrected_plate,
            violation="Character Manipulation",
            confidence_modifier=0.95,
        )

    # Priority 3: Spacing Manipulation
    # Flag ANY plate that arrived with spaces or hyphens
    if has_spacing:
        m = PLATE_PATTERN.match(corrected_plate)
        if m:
            formatted = f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"
        else:
            formatted = corrected_plate

        return PlateValidationResult(
            detected_plate=cleaned,
            correct_plate=formatted,
            violation="Spacing Manipulation",
            confidence_modifier=0.9,
        )

    # Priority 4: Pattern Mismatch (doesn't fit Indian format)
    if not PLATE_PATTERN.match(corrected_plate):
        return PlateValidationResult(
            detected_plate=cleaned,
            correct_plate=corrected_plate,
            violation="Plate Pattern Mismatch",
            confidence_modifier=0.8,
        )

    # Priority 5: Invalid State Code
    if state not in VALID_STATE_CODES:
        return PlateValidationResult(
            detected_plate=cleaned,
            correct_plate=corrected_plate,
            violation="Invalid State Code",
            confidence_modifier=0.7,
        )

    # No violations — valid plate
    return PlateValidationResult(
        detected_plate=cleaned,
        correct_plate=corrected_plate,
        violation=None,
    )
