"""
RoadVision — Indian Number Plate Format Rules
Validates detected plate text against Indian RTO formatting standards.
Detects character manipulation, spacing issues, and pattern mismatches.
Includes context-aware normalization that considers character position.
Integrates with vehicle registration database for ownership verification.
"""

import re
import logging
from typing import Optional, Dict
from registration_db import lookup_vehicle, is_registered_plate

logger = logging.getLogger(__name__)

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

# Common OCR letter-to-letter confusions (used for state code correction)
# Key = common OCR error, Value = list of likely intended characters
LETTER_CONFUSION = {
    'M': 'N',
    'N': 'M',
    'E': 'B',
    'F': 'P',
    'V': 'U',
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
        vehicle_info: Optional[Dict] = None,
        font_anomaly: bool = False,
    ):
        self.detected_plate = detected_plate
        # Only set correct_plate if it differs from detected (actual correction)
        if correct_plate and correct_plate != detected_plate:
            self.correct_plate = correct_plate
        else:
            self.correct_plate = None  # No correction needed
        self.violation = violation
        self.confidence_modifier = confidence_modifier
        self.vehicle_info = vehicle_info or {}
        self.font_anomaly = font_anomaly

    @property
    def is_violation(self) -> bool:
        return self.violation is not None

    def to_dict(self) -> dict:
        result = {
            "detected_plate": self.detected_plate,
            "correct_plate": self.correct_plate,  # None if no correction
            "violation": self.violation,
            "font_anomaly": self.font_anomaly,
        }
        if self.vehicle_info:
            result["vehicle_info"] = self.vehicle_info
        return result


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

    # State-code-specific correction (e.g., TM -> TN)
    if len(corrected) >= 2:
        state = "".join(corrected[:2])
        if state not in VALID_STATE_CODES:
            # Try to fix state code using common letter confusions
            s0, s1 = corrected[0], corrected[1]
            
            # Case 1: First char confused (e.g., M... -> N...)
            if s0 in LETTER_CONFUSION:
                test_state = LETTER_CONFUSION[s0] + s1
                if test_state in VALID_STATE_CODES:
                    corrected[0] = LETTER_CONFUSION[s0]
                    corrections.append((0, s0, corrected[0]))
                    state = test_state
            
            # Case 2: Second char confused (e.g., TM -> TN)
            if state not in VALID_STATE_CODES and s1 in LETTER_CONFUSION:
                test_state = s0 + LETTER_CONFUSION[s1]
                if test_state in VALID_STATE_CODES:
                    corrected[1] = LETTER_CONFUSION[s1]
                    corrections.append((1, s1, corrected[1]))

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
    
    Accepts both spaced and non-spaced formats as valid:
    - "TN57AD3604" (no spaces)
    - "TN 57 AD 3604" (standard spacing)
    - "TN57 AD3604" (partial spacing)
    
    Only flags truly unusual patterns like:
    - "T N57AD3604" (space within state code)
    - "TN5 7AD3604" (space within district code)
    - "TN57A D3604" (space within series)
    - "TN57AD36 04" (space within registration number)
    """
    # If the original text had no spaces, no spacing issue
    if ' ' not in original_text and '-' not in original_text:
        return None

    # Standard Indian spacing patterns (acceptable)
    # These patterns accept:
    # 1. Full spacing: "AA NN AA NNNN" or "AA NN AAA NNNN"
    # 2. No spacing: "AANNAAANNNN"
    # 3. Partial spacing: "AA NN AANNNN", "AANN AA NNNN", etc.
    # 4. With hyphens: "AA-NN-AA-NNNN"
    standard_patterns = [
        # Full format with optional spaces/hyphens between segments
        re.compile(r'^[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{1,4}$'),
        # Allow multiple spaces or hyphens
        re.compile(r'^[A-Z]{2}[\s\-]+\d{2}[\s\-]+[A-Z]{1,3}[\s\-]+\d{1,4}$'),
    ]

    cleaned = original_text.upper().strip()
    for pattern in standard_patterns:
        if pattern.match(cleaned):
            return None

    # Check for truly unusual spacing patterns (spaces WITHIN segments)
    # These are suspicious and should be flagged
    unusual_patterns = [
        r'[A-Z]\s+[A-Z](?=\d)',  # Space within state code: "T N57"
        r'\d\s+\d(?=[A-Z])',      # Space within district code: "5 7AD"
        r'(?<=\d)[A-Z]\s+[A-Z](?=\d)',  # Space within series: "A D3604"
        r'[A-Z]\d+\s+\d',         # Space within registration number: "AD36 04"
    ]
    
    for pattern in unusual_patterns:
        if re.search(pattern, cleaned):
            # Truly unusual spacing detected
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

    # If we reach here, spacing is non-standard but not suspicious
    # Accept it as valid
    return None


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


def detect_font_anomaly(plate_crop) -> dict:
    """
    Analyze the plate crop image for non-standard font characteristics.
    Indian regulation requires plates to use a specific standard font
    (Charles Wright / Mandatory font as per HSRP rules).
    Fancy/stylized/italic/decorative fonts are illegal.

    Uses STRICT thresholds to avoid false positives on standard plates
    captured via low-quality webcam images (which naturally have noisy
    edges, uneven lighting, and compression artifacts).

    Only flags truly decorative/fancy fonts — NOT standard plates
    that happen to have low image quality.

    Returns dict with:
      - is_anomaly: bool (True if non-standard font detected)
      - confidence: float (0-1, how confident we are it's non-standard)
      - reason: str (description of the anomaly)
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"is_anomaly": False, "confidence": 0.0, "reason": ""}

    if plate_crop is None or (hasattr(plate_crop, 'size') and plate_crop.size == 0):
        return {"is_anomaly": False, "confidence": 0.0, "reason": ""}

    anomaly_score = 0.0
    reasons = []

    # Convert to grayscale
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop.copy()

    h, w = gray.shape[:2]
    if h < 15 or w < 40:
        return {"is_anomaly": False, "confidence": 0.0, "reason": ""}

    # --- 1. Stroke Width Variance ---
    # Standard block fonts have very uniform stroke widths.
    # Only flag when CV is EXTREMELY high (fancy calligraphy/script fonts).
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_pixels = dist_transform[dist_transform > 0]

    if len(stroke_pixels) > 50:
        stroke_std = np.std(stroke_pixels)
        stroke_mean = np.mean(stroke_pixels)
        if stroke_mean > 0:
            cv_ratio = stroke_std / stroke_mean
            # Very strict: only flag CV > 0.85 (script/calligraphy fonts)
            if cv_ratio > 0.85:
                anomaly_score += 0.30
                reasons.append(f"Highly variable stroke width (CV={cv_ratio:.2f})")

    # --- 2. Edge Density ---
    # Only flag extremely dense edges (ornamental/decorative fonts).
    # Standard plates with noise typically stay under 0.30.
    edges = cv2.Canny(gray, 80, 200)  # Tighter Canny thresholds
    edge_density = np.sum(edges > 0) / (h * w)
    if edge_density > 0.35:
        anomaly_score += 0.20
        reasons.append(f"Very high edge density ({edge_density:.2f}) — decorative font")

    # --- 3. Character Contour Analysis ---
    # Only flag when characters are extremely jagged/ornate.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) >= 4:
        complexity_scores = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if area > 50 and perimeter > 0:  # Larger minimum area
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                complexity_scores.append(circularity)

        if len(complexity_scores) >= 3:
            avg_circularity = np.mean(complexity_scores)
            # Very strict: only flag circularity < 0.10 (heavily ornate)
            if avg_circularity < 0.10:
                anomaly_score += 0.25
                reasons.append(f"Highly ornate character shapes (circularity={avg_circularity:.2f})")

    # --- 4. Vertical Projection Regularity ---
    # Only flag extreme spacing irregularity (CV > 1.2).
    v_projection = np.sum(binary > 0, axis=0)
    threshold = h * 0.15
    gaps = []
    in_gap = False
    gap_start = 0
    for col_idx, val in enumerate(v_projection):
        if val < threshold and not in_gap:
            in_gap = True
            gap_start = col_idx
        elif val >= threshold and in_gap:
            in_gap = False
            gap_width = col_idx - gap_start
            if gap_width > 2:  # Ignore tiny gaps (noise)
                gaps.append(gap_width)

    if len(gaps) >= 4:
        gap_std = np.std(gaps)
        gap_mean = np.mean(gaps)
        if gap_mean > 0:
            gap_cv = gap_std / gap_mean
            if gap_cv > 1.2:
                anomaly_score += 0.15
                reasons.append(f"Very irregular character spacing (gap CV={gap_cv:.2f})")

    # Final decision — STRICT trigger threshold
    # Requires multiple strong signals to flag as non-standard font
    is_anomaly = anomaly_score >= 0.55
    confidence = min(anomaly_score, 1.0)
    reason = "; ".join(reasons) if reasons else ""

    if is_anomaly:
        logger.info(f"Font anomaly detected (score={anomaly_score:.2f}): {reason}")
    else:
        logger.debug(f"Font check passed (score={anomaly_score:.2f})")

    return {
        "is_anomaly": is_anomaly,
        "confidence": confidence,
        "reason": reason,
    }


def validate_plate(raw_text: str, plate_crop=None) -> PlateValidationResult:
    """
    Run all validation rules on a detected plate.

    Only plates that have a violation get a 'correct_plate' reconstruction.
    Legal plates are left as-is (correct_plate = None = no correction needed).

    Detection order (violations checked BEFORE normalization hides them):
      1. Store the original OCR output
      2. Detect spacing manipulation on the raw text (before cleaning)
      3. Apply character normalization (I→1, O→0, B→8, S→5, Z→2)
      4. Compare original vs normalized → Character Manipulation
      5. Validate normalized plate against Indian regex
      6. Check vehicle registration database
      7. Return both detected and corrected plates with vehicle info
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

    # Step 3c: Detect font anomalies on the plate crop image
    font_result = detect_font_anomaly(plate_crop) if plate_crop is not None else {"is_anomaly": False, "confidence": 0.0, "reason": ""}
    has_font_anomaly = font_result.get("is_anomaly", False)

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
            font_anomaly=has_font_anomaly,
        )

    # Priority 2: Character Manipulation (original differs from corrected)
    if corrections:
        violation_type = "Non-Standard Font / Character Manipulation" if has_font_anomaly else "Character Manipulation"
        return PlateValidationResult(
            detected_plate=cleaned,
            correct_plate=corrected_plate,
            violation=violation_type,
            confidence_modifier=0.90 if has_font_anomaly else 0.95,
            font_anomaly=has_font_anomaly,
        )

    # Priority 3: Spacing Manipulation
    # Only flag plates with truly unusual spacing patterns
    if has_spacing:
        spacing_result = check_spacing_manipulation(original_plate, corrected_plate)
        if spacing_result:
            return spacing_result

    # Priority 4: Non-Standard Font (without other violations)
    if has_font_anomaly and not corrections:
        return PlateValidationResult(
            detected_plate=cleaned,
            correct_plate=None,  # No text correction needed, just font flagged
            violation="Non-Standard Font",
            confidence_modifier=0.85,
            font_anomaly=True,
        )

    # Priority 5: Pattern Mismatch (doesn't fit Indian format)
    if not PLATE_PATTERN.match(corrected_plate):
        return PlateValidationResult(
            detected_plate=cleaned,
            correct_plate=corrected_plate,
            violation="Plate Pattern Mismatch",
            confidence_modifier=0.8,
            font_anomaly=has_font_anomaly,
        )

    # Priority 5: Check vehicle registration database
    # Look up the corrected plate (after all format fixes)
    # This check happens BEFORE invalid state code check because
    # an unregistered vehicle is a more serious violation
    vehicle_info = lookup_vehicle(corrected_plate)
    logger.info(f"Registration lookup for '{corrected_plate}' (cleaned='{cleaned}'): registered={vehicle_info.get('registered')}")
    
    if not vehicle_info.get("registered"):
        # Unregistered vehicle - this is a violation
        logger.info(f"Unregistered vehicle detected: '{corrected_plate}'")
        return PlateValidationResult(
            detected_plate=cleaned,
            correct_plate=corrected_plate if corrected_plate != cleaned else None,
            violation="Unregistered Vehicle",
            confidence_modifier=0.85,
            vehicle_info=vehicle_info,
            font_anomaly=has_font_anomaly,
        )

    # Priority 6: Invalid State Code (only if registered check passed)
    # This is now lower priority since we already checked registration
    if state not in VALID_STATE_CODES:
        return PlateValidationResult(
            detected_plate=cleaned,
            correct_plate=corrected_plate if corrected_plate != cleaned else None,
            violation="Invalid State Code",
            confidence_modifier=0.7,
            vehicle_info=vehicle_info,
            font_anomaly=has_font_anomaly,
        )

    # No violations — valid and registered plate
    # Legal plates get NO correction (correct_plate = None)
    return PlateValidationResult(
        detected_plate=cleaned,
        correct_plate=None,  # No correction needed for legal plates
        violation=None,
        vehicle_info=vehicle_info,
        font_anomaly=has_font_anomaly,
    )
