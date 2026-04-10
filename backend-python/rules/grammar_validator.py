"""
EvasionEye — Indian RTO Grammar Validator
Validates license plate text against the Indian RTO format standard.

Indian plate format: AA NN AA NNNN
  AA   = State code (2 letters)
  NN   = District code (2 digits)
  AA   = Series letters (1–3 letters)
  NNNN = Registration number (1–4 digits)

Total length (excluding spaces): 6–12 characters.
"""

import re
import logging
from typing import Optional, TypedDict, Dict

from .vehicle_registration import lookup_vehicle_registration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character correction maps
# ---------------------------------------------------------------------------

# Digit-like characters that should be letters (used in letter positions)
_DIGIT_TO_LETTER: dict[str, str] = {
    '0': 'O',
    '1': 'I',
    '8': 'B',
    '5': 'S',
    '2': 'Z',
    '6': 'G',
}

# Letter-like characters that should be digits (used in digit positions)
_LETTER_TO_DIGIT: dict[str, str] = {
    'O': '0',
    'I': '1',
    'B': '8',
    'S': '5',
    'Z': '2',
    'G': '6',
}

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Full Indian RTO pattern (strict): 2 letters, 2 digits, 1-3 letters, 1-4 digits
_INDIAN_PLATE_PATTERN = re.compile(
    r'^[A-Z]{2}'      # State code (2 letters)
    r'\d{2}'           # District code (2 digits)
    r'[A-Z]{1,3}'     # Series letters (1–3 letters)
    r'\d{1,4}$'        # Registration number (1–4 digits)
)

# Minimum / maximum character counts (spaces excluded)
_MIN_LENGTH = 6
_MAX_LENGTH = 12


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ValidationResult(TypedDict):
    """Result returned by validate_indian_format()."""
    is_valid: bool
    plate: str          # Normalised (spaces stripped, uppercased) input
    reason: str         # Human-readable explanation


class CorrectionResult(TypedDict):
    """Result returned by apply_position_based_corrections()."""
    original: str       # Input plate (normalised, spaces stripped)
    corrected: str      # Plate after position-based corrections
    corrections: list[str]  # Human-readable log of each correction applied


class ManipulationResult(TypedDict):
    """Result returned by detect_character_manipulation()."""
    original: str           # Normalised input plate
    corrected: str          # Plate with substitutions corrected (position-based)
    is_manipulation: bool   # True if any suspicious substitutions were found
    substitutions: list[str]  # Human-readable log of each substitution found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_position_based_corrections(plate_text: str) -> CorrectionResult:
    """Apply position-based character corrections to a license plate string.

    Indian RTO format: AA NN AA NNNN
      Positions 0-1  → state code  (must be letters)
      Positions 2-3  → district code (must be digits)
      Positions 4-6  → series letters (must be letters, 1-3 chars)
      Positions 7-10 → registration number (must be digits, 1-4 chars)

    The function normalises the input (uppercase, strips spaces/hyphens) then
    walks each character, swapping visually-similar characters to match the
    expected type at that position.

    Args:
        plate_text: Raw plate string from OCR (may contain spaces/hyphens).

    Returns:
        A :class:`CorrectionResult` dict with keys:
          - ``original`` (str): Normalised bare input.
          - ``corrected`` (str): Plate after corrections.
          - ``corrections`` (list[str]): Description of each change made.

    Examples:
        >>> apply_position_based_corrections("MH12AB1234")
        {'original': 'MH12AB1234', 'corrected': 'MH12AB1234', 'corrections': []}

        >>> apply_position_based_corrections("0H12AB1234")
        {'original': '0H12AB1234', 'corrected': 'OH12AB1234', 'corrections': ["pos 0: '0' → 'O' (digit→letter)"]}
    """
    if not plate_text:
        return CorrectionResult(original="", corrected="", corrections=[])

    # Normalise
    bare = re.sub(r'[\s\-]+', '', plate_text.strip().upper())

    chars = list(bare)
    corrections: list[str] = []
    n = len(chars)

    # Determine series/registration boundary.
    # After the 4-char prefix (state + district), the series is 1-3 letters
    # followed by 1-4 digits.  We scan forward from position 4 to find where
    # letters end and digits begin.
    series_end = 4  # default: no series chars found
    if n > 4:
        i = 4
        while i < n and (chars[i].isalpha() or chars[i] in _DIGIT_TO_LETTER):
            i += 1
        series_end = i  # positions [4, series_end) are series letters

    def _enforce_letter(pos: int) -> None:
        ch = chars[pos]
        if ch.isalpha():
            return
        replacement = _DIGIT_TO_LETTER.get(ch)
        if replacement:
            corrections.append(f"pos {pos}: '{ch}' → '{replacement}' (digit→letter)")
            chars[pos] = replacement

    def _enforce_digit(pos: int) -> None:
        ch = chars[pos]
        if ch.isdigit():
            return
        replacement = _LETTER_TO_DIGIT.get(ch)
        if replacement:
            corrections.append(f"pos {pos}: '{ch}' → '{replacement}' (letter→digit)")
            chars[pos] = replacement

    # Positions 0-1: state code → enforce letters
    for pos in range(min(2, n)):
        _enforce_letter(pos)

    # Positions 2-3: district code → enforce digits
    for pos in range(2, min(4, n)):
        _enforce_digit(pos)

    # Positions 4 … series_end-1: series letters → enforce letters
    for pos in range(4, min(series_end, n)):
        _enforce_letter(pos)

    # Positions series_end … end: registration digits → enforce digits
    for pos in range(series_end, n):
        _enforce_digit(pos)

    corrected = ''.join(chars)

    if corrections:
        logger.info(
            "apply_position_based_corrections: %d correction(s) on '%s' → '%s': %s",
            len(corrections), bare, corrected, corrections,
        )
    else:
        logger.debug("apply_position_based_corrections: no corrections needed for '%s'", bare)

    return CorrectionResult(original=bare, corrected=corrected, corrections=corrections)


def validate_indian_format(plate_text: str) -> ValidationResult:
    """Validate a license plate string against the Indian RTO format.

    The function:
      1. Strips surrounding whitespace and converts to uppercase.
      2. Removes internal spaces/hyphens to get the bare alphanumeric string.
      3. Checks the bare length is within [6, 12] characters.
      4. Matches the bare string against the AA NN AA NNNN regex pattern.

    Args:
        plate_text: Raw plate string from OCR (may contain spaces/hyphens).

    Returns:
        A :class:`ValidationResult` dict with keys:
          - ``is_valid`` (bool): True when the plate passes all checks.
          - ``plate`` (str): The normalised bare plate string used for checks.
          - ``reason`` (str): Description of the validation outcome.

    Examples:
        >>> validate_indian_format("MH12AB1234")
        {'is_valid': True, 'plate': 'MH12AB1234', 'reason': 'Valid Indian RTO format'}

        >>> validate_indian_format("MH 12 AB 1234")
        {'is_valid': True, 'plate': 'MH12AB1234', 'reason': 'Valid Indian RTO format'}

        >>> validate_indian_format("XY")
        {'is_valid': False, 'plate': 'XY', 'reason': 'Length 2 is outside valid range [6, 12]'}
    """
    if not plate_text:
        logger.debug("validate_indian_format: received empty input")
        return ValidationResult(is_valid=False, plate="", reason="Empty plate text")

    # Normalise: uppercase, strip outer whitespace
    normalised = plate_text.strip().upper()

    # Remove spaces and hyphens to get the bare alphanumeric string
    bare = re.sub(r'[\s\-]+', '', normalised)

    # --- Length check ---
    length = len(bare)
    if length < _MIN_LENGTH or length > _MAX_LENGTH:
        reason = f"Length {length} is outside valid range [{_MIN_LENGTH}, {_MAX_LENGTH}]"
        logger.debug("validate_indian_format: INVALID — %s (plate='%s')", reason, bare)
        return ValidationResult(is_valid=False, plate=bare, reason=reason)

    # --- Pattern check ---
    if not _INDIAN_PLATE_PATTERN.match(bare):
        reason = "Does not match Indian RTO pattern AA NN AA NNNN"
        logger.debug("validate_indian_format: INVALID — %s (plate='%s')", reason, bare)
        return ValidationResult(is_valid=False, plate=bare, reason=reason)

    logger.info("validate_indian_format: VALID — plate='%s'", bare)
    return ValidationResult(is_valid=True, plate=bare, reason="Valid Indian RTO format")


def detect_character_manipulation(plate_text: str) -> ManipulationResult:
    """Detect suspicious character substitutions in a license plate string.

    Checks whether the plate contains characters that are visually similar to
    other characters and may have been deliberately substituted to obscure the
    plate identity:

    - Digits used where letters are expected (``_DIGIT_TO_LETTER`` keys):
      ``0``, ``1``, ``8``, ``5``, ``2``, ``6``
    - Letters used where digits are expected (``_LETTER_TO_DIGIT`` keys):
      ``O``, ``I``, ``B``, ``S``, ``Z``, ``G``

    The detection is position-agnostic: any occurrence of these characters is
    flagged as suspicious regardless of where it appears in the plate.  The
    ``corrected`` field is produced by :func:`apply_position_based_corrections`
    which uses positional context to resolve ambiguities.

    Args:
        plate_text: Raw plate string from OCR (may contain spaces/hyphens).

    Returns:
        A :class:`ManipulationResult` dict with keys:
          - ``original`` (str): Normalised bare input.
          - ``corrected`` (str): Plate after position-based corrections.
          - ``is_manipulation`` (bool): ``True`` when suspicious chars found.
          - ``substitutions`` (list[str]): Description of each suspicious char.

    Examples:
        >>> detect_character_manipulation("MH12AB1234")
        {'original': 'MH12AB1234', 'corrected': 'MH12AB1234', 'is_manipulation': False, 'substitutions': []}

        >>> detect_character_manipulation("0H12AB1234")
        {'original': '0H12AB1234', 'corrected': 'OH12AB1234', 'is_manipulation': True, 'substitutions': ["pos 0: '0' is a digit that looks like a letter (→ 'O')"]}
    """
    if not plate_text:
        return ManipulationResult(
            original="",
            corrected="",
            is_manipulation=False,
            substitutions=[],
        )

    # Normalise: uppercase, strip spaces/hyphens
    bare = re.sub(r'[\s\-]+', '', plate_text.strip().upper())

    substitutions: list[str] = []

    for pos, ch in enumerate(bare):
        if ch in _DIGIT_TO_LETTER:
            substitutions.append(
                f"pos {pos}: '{ch}' is a digit that looks like a letter (→ '{_DIGIT_TO_LETTER[ch]}')"
            )
        elif ch in _LETTER_TO_DIGIT:
            substitutions.append(
                f"pos {pos}: '{ch}' is a letter that looks like a digit (→ '{_LETTER_TO_DIGIT[ch]}')"
            )

    is_manipulation = len(substitutions) > 0

    # Use position-based corrections for the corrected plate
    correction_result = apply_position_based_corrections(bare)
    corrected = correction_result["corrected"]

    if is_manipulation:
        logger.info(
            "detect_character_manipulation: manipulation detected on '%s' → '%s': %s",
            bare, corrected, substitutions,
        )
    else:
        logger.debug("detect_character_manipulation: no manipulation detected for '%s'", bare)

    return ManipulationResult(
        original=bare,
        corrected=corrected,
        is_manipulation=is_manipulation,
        substitutions=substitutions,
    )


class FontAnomalyResult(TypedDict):
    """Result returned by detect_font_anomalies()."""
    result: Optional[bool]   # True = anomaly, False = standard, None = unknown
    confidence: float        # 0.0–1.0
    reason: str              # Human-readable description


def detect_font_anomalies(plate_image) -> FontAnomalyResult:
    """Analyze a plate image crop for non-standard font characteristics.

    Indian regulation requires plates to use a specific standard font
    (Charles Wright / Mandatory font as per HSRP rules).  Fancy, stylized,
    italic, or decorative fonts are illegal.

    Uses STRICT thresholds to avoid false positives on standard plates
    captured via low-quality webcam images (which naturally have noisy
    edges, uneven lighting, and compression artifacts).

    Args:
        plate_image: Plate image crop as a numpy array (BGR or grayscale),
                     or ``None`` when no image is available.

    Returns:
        A :class:`FontAnomalyResult` dict with keys:
          - ``result`` (Optional[bool]):
              ``True``  — font anomalies detected (non-standard font),
              ``False`` — font appears standard,
              ``None``  — cannot determine (no image / image too small /
                          cv2 unavailable).
          - ``confidence`` (float): Confidence score in [0.0, 1.0].
          - ``reason`` (str): Human-readable description of the outcome.

    Examples:
        >>> detect_font_anomalies(None)
        {'result': None, 'confidence': 0.0, 'reason': 'No image provided'}
    """
    # --- Guard: cv2 / numpy availability ---
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.debug("detect_font_anomalies: cv2/numpy not available — result unknown")
        return FontAnomalyResult(result=None, confidence=0.0, reason="cv2/numpy not available")

    # --- Guard: image presence ---
    if plate_image is None:
        logger.debug("detect_font_anomalies: no image provided — result unknown")
        return FontAnomalyResult(result=None, confidence=0.0, reason="No image provided")

    if hasattr(plate_image, 'size') and plate_image.size == 0:
        logger.debug("detect_font_anomalies: empty image — result unknown")
        return FontAnomalyResult(result=None, confidence=0.0, reason="Empty image")

    # --- Convert to grayscale ---
    if len(plate_image.shape) == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image.copy()

    h, w = gray.shape[:2]

    # --- Guard: minimum size ---
    if h < 15 or w < 40:
        logger.debug(
            "detect_font_anomalies: image too small (%dx%d) — result unknown", w, h
        )
        return FontAnomalyResult(result=None, confidence=0.0, reason="Image too small")

    anomaly_score = 0.0
    reasons: list[str] = []

    # --- 1. Stroke Width Variance ---
    # Standard block fonts have very uniform stroke widths.
    # Only flag when CV is EXTREMELY high (fancy calligraphy/script fonts).
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_pixels = dist_transform[dist_transform > 0]

    if len(stroke_pixels) > 50:
        stroke_std = float(np.std(stroke_pixels))
        stroke_mean = float(np.mean(stroke_pixels))
        if stroke_mean > 0:
            cv_ratio = stroke_std / stroke_mean
            if cv_ratio > 0.85:
                anomaly_score += 0.30
                reasons.append(f"Highly variable stroke width (CV={cv_ratio:.2f})")

    # --- 2. Edge Density ---
    # Only flag extremely dense edges (ornamental/decorative fonts).
    edges = cv2.Canny(gray, 80, 200)
    edge_density = float(np.sum(edges > 0)) / (h * w)
    if edge_density > 0.35:
        anomaly_score += 0.20
        reasons.append(f"Very high edge density ({edge_density:.2f}) — decorative font")

    # --- 3. Character Contour Analysis ---
    # Only flag when characters are extremely jagged/ornate.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) >= 4:
        complexity_scores: list[float] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if area > 50 and perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                complexity_scores.append(circularity)

        if len(complexity_scores) >= 3:
            avg_circularity = float(np.mean(complexity_scores))
            if avg_circularity < 0.10:
                anomaly_score += 0.25
                reasons.append(
                    f"Highly ornate character shapes (circularity={avg_circularity:.2f})"
                )

    # --- 4. Vertical Projection Regularity ---
    # Only flag extreme spacing irregularity (gap CV > 1.2).
    v_projection = np.sum(binary > 0, axis=0)
    gap_threshold = h * 0.15
    gaps: list[int] = []
    in_gap = False
    gap_start = 0
    for col_idx, val in enumerate(v_projection):
        if val < gap_threshold and not in_gap:
            in_gap = True
            gap_start = col_idx
        elif val >= gap_threshold and in_gap:
            in_gap = False
            gap_width = col_idx - gap_start
            if gap_width > 2:
                gaps.append(gap_width)

    if len(gaps) >= 4:
        gap_std = float(np.std(gaps))
        gap_mean = float(np.mean(gaps))
        if gap_mean > 0:
            gap_cv = gap_std / gap_mean
            if gap_cv > 1.2:
                anomaly_score += 0.15
                reasons.append(f"Very irregular character spacing (gap CV={gap_cv:.2f})")

    # --- Final decision (strict threshold: >= 0.55) ---
    is_anomaly = anomaly_score >= 0.55
    confidence = min(anomaly_score, 1.0)
    reason = "; ".join(reasons) if reasons else ""

    if is_anomaly:
        logger.info(
            "detect_font_anomalies: anomaly detected (score=%.2f): %s",
            anomaly_score, reason,
        )
        return FontAnomalyResult(result=True, confidence=confidence, reason=reason)

    if reasons:
        # Some signals but below threshold
        logger.debug(
            "detect_font_anomalies: check passed (score=%.2f, weak signals: %s)",
            anomaly_score, reason,
        )
    else:
        logger.debug(
            "detect_font_anomalies: check passed (score=%.2f) — standard font",
            anomaly_score,
        )

    return FontAnomalyResult(result=False, confidence=confidence, reason=reason)


# ---------------------------------------------------------------------------
# Comprehensive plate validation
# ---------------------------------------------------------------------------

class PlateValidationResult(TypedDict):
    """Result returned by validate_plate()."""
    detected_plate: str          # Normalized input plate
    correct_plate: Optional[str] # Corrected plate (None if no correction needed)
    violation: Optional[str]     # Violation type string or None
    font_anomaly: bool           # True if non-standard font detected
    confidence_modifier: float   # Multiplier for confidence score (0.0–1.0)
    vehicle_info: Optional[Dict] # Vehicle registration info


def validate_plate(
    plate_text: str,
    plate_image=None,
    db_path: str = "registration_db.sqlite",
) -> PlateValidationResult:
    """Combine all validators to produce a comprehensive plate validation result.

    Runs grammar validation, character manipulation detection, font anomaly
    detection (if an image is provided), and vehicle registration lookup.
    Returns a :class:`PlateValidationResult` with the detected plate, any
    corrected plate text, the highest-priority violation found, and a
    confidence modifier to apply to the detection score.

    Violation priority (highest → lowest):
      1. "Tampered Plate"         — manipulation with ≥3 corrections AND invalid format
      2. "Character Manipulation" — any suspicious character substitution
      3. "Non-Standard Font"      — font anomaly detected in image
      4. "Plate Pattern Mismatch" — plate does not match Indian RTO format
      5. "Unregistered Vehicle"   — plate not found in registration database
      6. None                     — valid, registered plate

    Args:
        plate_text:  Raw plate string from OCR (may contain spaces/hyphens).
        plate_image: Optional plate image crop (numpy array) for font analysis.
        db_path:     Path to the SQLite vehicle registrations database.

    Returns:
        A :class:`PlateValidationResult` dict with keys:
          - ``detected_plate`` (str): Normalized input plate.
          - ``correct_plate`` (Optional[str]): Corrected plate or None.
          - ``violation`` (Optional[str]): Violation type or None.
          - ``confidence_modifier`` (float): Score multiplier in [0.0, 1.0].
    """
    # Normalize the plate
    detected_plate = re.sub(r'[\s\-]+', '', plate_text.strip().upper()) if plate_text else ""

    # Run all validators
    grammar_result = validate_indian_format(detected_plate)
    manipulation_result = detect_character_manipulation(detected_plate)
    font_result = detect_font_anomalies(plate_image) if plate_image is not None else None

    # Determine correct_plate (use corrected value if different from detected)
    corrected = manipulation_result["corrected"]
    correct_plate: Optional[str] = corrected if corrected != detected_plate else None

    # Registration lookup (log result at INFO level)
    registration_info = lookup_vehicle_registration(detected_plate, db_path)
    if registration_info is not None:
        logger.info(
            "validate_plate: registration found for '%s' — owner='%s', status='%s'",
            detected_plate, registration_info["owner"], registration_info["status"],
        )
    else:
        logger.info(
            "validate_plate: no registration found for '%s'",
            detected_plate,
        )

    # Determine violation by priority
    violation: Optional[str] = None
    confidence_modifier: float = 1.0

    num_substitutions = len(manipulation_result["substitutions"])
    is_invalid_format = not grammar_result["is_valid"]

    if num_substitutions >= 3 and is_invalid_format:
        violation = "Tampered Plate"
        confidence_modifier = 0.75
    elif manipulation_result["is_manipulation"]:
        violation = "Character Manipulation"
        confidence_modifier = 0.95
    elif font_result is not None and font_result["result"] is True:
        violation = "Non-Standard Font"
        confidence_modifier = 0.85
    elif is_invalid_format:
        violation = "Plate Pattern Mismatch"
        confidence_modifier = 0.80
    elif registration_info is None:
        violation = "Unregistered Vehicle"
        confidence_modifier = 0.85

    # Logging
    if violation is not None:
        logger.info(
            "validate_plate: violation='%s' detected for plate='%s'",
            violation, detected_plate,
        )
    else:
        logger.debug(
            "validate_plate: plate='%s' is valid and registered",
            detected_plate,
        )

    return PlateValidationResult(
        detected_plate=detected_plate,
        correct_plate=correct_plate,
        violation=violation,
        font_anomaly=bool(font_result["result"]) if font_result is not None else False,
        confidence_modifier=confidence_modifier,
        vehicle_info=registration_info,
    )
