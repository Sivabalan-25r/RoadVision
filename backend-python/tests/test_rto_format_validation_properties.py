"""
Property-Based Tests for RTO Format Validation

This module contains property-based tests that verify validate_plate()
correctly handles RTO format validation as part of comprehensive plate
validation.

**Property 48: RTO Format Validation**
**Validates: Requirements 19.1, 19.2, 19.3**

## Requirements

Requirement 19.1: The system shall not assign a "Plate Pattern Mismatch"
violation to plates that conform to the Indian RTO format (AA NN AA NNNN).

Requirement 19.2: The system shall assign a "Plate Pattern Mismatch"
violation to plates that do not conform to the Indian RTO format, provided
no higher-priority violation (character manipulation) is present.

Requirement 19.3: validate_plate() shall always return a PlateValidationResult
with the required fields: detected_plate, correct_plate, violation,
confidence_modifier.

## Properties Tested

- **Property 48a**: Valid RTO format plates (pure chars, no manipulation) do
  NOT receive a "Plate Pattern Mismatch" violation.

- **Property 48b**: Invalid format plates with no ambiguous characters receive
  a "Plate Pattern Mismatch" violation.

- **Property 48c**: PlateValidationResult always contains required fields with
  correct types for any input.

- **Property 48d**: confidence_modifier is always in [0.0, 1.0].

- **Property 48e**: detected_plate is always the normalised (stripped,
  uppercased, spaces/hyphens removed) form of the input.
"""

import os
import re
import sqlite3
import sys
import tempfile

import pytest
from hypothesis import given, settings, strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.grammar_validator import validate_plate

# ---------------------------------------------------------------------------
# Character sets
# ---------------------------------------------------------------------------

# Substitution digits/letters that trigger character manipulation detection
_SUBSTITUTION_DIGITS = set("018526")
_SUBSTITUTION_LETTERS = set("OIBSZG")
_AMBIGUOUS_CHARS = _SUBSTITUTION_DIGITS | _SUBSTITUTION_LETTERS

# Pure chars: no ambiguous characters
_PURE_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - _SUBSTITUTION_LETTERS
_PURE_DIGITS = set("0123456789") - _SUBSTITUTION_DIGITS

_PURE_ALPHA = "".join(sorted(_PURE_LETTERS))   # e.g. "ACDEFHJKLMNPQRTUVWXY"
_PURE_DIGIT = "".join(sorted(_PURE_DIGITS))    # e.g. "3479"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_empty_db() -> str:
    """Create a temporary empty SQLite DB (no registrations)."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE vehicle_registrations (
            plate_number      TEXT PRIMARY KEY,
            owner             TEXT NOT NULL,
            registration_date TEXT NOT NULL,
            status            TEXT NOT NULL DEFAULT 'active'
        )
        """
    )
    conn.commit()
    conn.close()
    return path


def _normalise(plate_text: str) -> str:
    """Mirror the normalisation applied inside validate_plate()."""
    return re.sub(r'[\s\-]+', '', plate_text.strip().upper())


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def valid_rto_plate_pure(draw):
    """
    Generate a syntactically valid Indian RTO plate using only pure chars.

    Format: AA NN AA NNNN  (no spaces in output)
      AA   = 2 pure letters (state code)
      NN   = 2 pure digits  (district code)
      AA   = 1–3 pure letters (series)
      NNNN = 1–4 pure digits  (registration)
    """
    state = draw(st.text(alphabet=_PURE_ALPHA, min_size=2, max_size=2))
    district = draw(st.text(alphabet=_PURE_DIGIT, min_size=2, max_size=2))
    series = draw(st.text(alphabet=_PURE_ALPHA, min_size=1, max_size=3))
    reg = draw(st.text(alphabet=_PURE_DIGIT, min_size=1, max_size=4))
    return state + district + series + reg


@st.composite
def invalid_format_plate_pure(draw):
    """
    Generate a plate that does NOT match the Indian RTO pattern and contains
    only pure characters (no ambiguous chars), so no manipulation is detected.

    Strategies to produce invalid plates:
      - Too short (< 6 chars)
      - Too long (> 12 chars)
      - Wrong structure (e.g. starts with digits, all letters, all digits)
    """
    strategy = draw(st.integers(min_value=0, max_value=2))

    if strategy == 0:
        # Too short: 1–5 pure chars
        return draw(st.text(
            alphabet=_PURE_ALPHA + _PURE_DIGIT,
            min_size=1,
            max_size=5,
        ))
    elif strategy == 1:
        # Too long: 13–20 pure chars
        return draw(st.text(
            alphabet=_PURE_ALPHA + _PURE_DIGIT,
            min_size=13,
            max_size=20,
        ))
    else:
        # Wrong structure: starts with digits (invalid for state code)
        leading_digits = draw(st.text(alphabet=_PURE_DIGIT, min_size=2, max_size=2))
        rest = draw(st.text(
            alphabet=_PURE_ALPHA + _PURE_DIGIT,
            min_size=4,
            max_size=8,
        ))
        return leading_digits + rest


# ---------------------------------------------------------------------------
# Property 48: RTO Format Validation
# ---------------------------------------------------------------------------

class TestRTOFormatValidationProperties:
    """
    Property-based tests for validate_plate() RTO format validation.

    **Property 48: RTO Format Validation**
    **Validates: Requirements 19.1, 19.2, 19.3**
    """

    # ------------------------------------------------------------------
    # Property 48a — valid RTO plates do NOT get "Plate Pattern Mismatch"
    # ------------------------------------------------------------------

    @given(plate=valid_rto_plate_pure())
    @settings(max_examples=200, deadline=5000)
    def test_property_48a_valid_rto_plate_no_pattern_mismatch(self, plate: str):
        """
        **Property 48a: Valid RTO format plates do not receive Plate Pattern Mismatch**

        For any plate that conforms to the Indian RTO format (AA NN AA NNNN)
        and contains only pure characters (no ambiguous substitution chars),
        validate_plate() SHALL NOT assign a "Plate Pattern Mismatch" violation.

        **Validates: Requirement 19.1**
        """
        db_path = _make_empty_db()
        try:
            result = validate_plate(plate, plate_image=None, db_path=db_path)
            assert result["violation"] != "Plate Pattern Mismatch", (
                f"Valid RTO plate '{plate}' must not receive 'Plate Pattern Mismatch', "
                f"got violation='{result['violation']}'"
            )
        finally:
            os.unlink(db_path)

    # ------------------------------------------------------------------
    # Property 48b — invalid format plates (pure chars) get "Plate Pattern Mismatch"
    # ------------------------------------------------------------------

    @given(plate=invalid_format_plate_pure())
    @settings(max_examples=200, deadline=5000)
    def test_property_48b_invalid_format_plate_gets_pattern_mismatch(self, plate: str):
        """
        **Property 48b: Invalid format plates receive Plate Pattern Mismatch violation**

        For any plate that does NOT conform to the Indian RTO format and
        contains only pure characters (no ambiguous chars that would trigger
        character manipulation detection), validate_plate() SHALL assign a
        "Plate Pattern Mismatch" violation.

        **Validates: Requirement 19.2**
        """
        db_path = _make_empty_db()
        try:
            result = validate_plate(plate, plate_image=None, db_path=db_path)
            assert result["violation"] == "Plate Pattern Mismatch", (
                f"Invalid format plate '{plate}' must receive 'Plate Pattern Mismatch', "
                f"got violation='{result['violation']}'"
            )
        finally:
            os.unlink(db_path)

    # ------------------------------------------------------------------
    # Property 48c — result always has required fields with correct types
    # ------------------------------------------------------------------

    @given(plate=st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_size=0,
        max_size=15,
    ))
    @settings(max_examples=300, deadline=5000)
    def test_property_48c_result_always_has_required_fields(self, plate: str):
        """
        **Property 48c: PlateValidationResult always contains required fields**

        validate_plate() SHALL always return a dict with:
          - detected_plate (str)
          - correct_plate (str or None)
          - violation (str or None)
          - confidence_modifier (float)

        **Validates: Requirement 19.3**
        """
        db_path = _make_empty_db()
        try:
            result = validate_plate(plate, plate_image=None, db_path=db_path)

            assert "detected_plate" in result, "Result must contain 'detected_plate'"
            assert "correct_plate" in result, "Result must contain 'correct_plate'"
            assert "violation" in result, "Result must contain 'violation'"
            assert "confidence_modifier" in result, "Result must contain 'confidence_modifier'"

            assert isinstance(result["detected_plate"], str), \
                "detected_plate must be str"
            assert result["correct_plate"] is None or isinstance(result["correct_plate"], str), \
                "correct_plate must be str or None"
            assert result["violation"] is None or isinstance(result["violation"], str), \
                "violation must be str or None"
            assert isinstance(result["confidence_modifier"], float), \
                "confidence_modifier must be float"
        finally:
            os.unlink(db_path)

    # ------------------------------------------------------------------
    # Property 48d — confidence_modifier is always in [0.0, 1.0]
    # ------------------------------------------------------------------

    @given(plate=st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_size=0,
        max_size=15,
    ))
    @settings(max_examples=300, deadline=5000)
    def test_property_48d_confidence_modifier_in_valid_range(self, plate: str):
        """
        **Property 48d: confidence_modifier is always in [0.0, 1.0]**

        For any input, the confidence_modifier returned by validate_plate()
        SHALL be a float in the range [0.0, 1.0].

        **Validates: Requirement 19.3**
        """
        db_path = _make_empty_db()
        try:
            result = validate_plate(plate, plate_image=None, db_path=db_path)
            modifier = result["confidence_modifier"]

            assert 0.0 <= modifier <= 1.0, (
                f"Plate '{plate}': confidence_modifier={modifier} is outside [0.0, 1.0]"
            )
        finally:
            os.unlink(db_path)

    # ------------------------------------------------------------------
    # Property 48e — detected_plate is always the normalised input
    # ------------------------------------------------------------------

    @given(plate=st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -",
        min_size=0,
        max_size=15,
    ))
    @settings(max_examples=300, deadline=5000)
    def test_property_48e_detected_plate_is_normalised_input(self, plate: str):
        """
        **Property 48e: detected_plate equals the normalised input**

        validate_plate() SHALL store the normalised plate (stripped, uppercased,
        spaces/hyphens removed) in the detected_plate field.

        **Validates: Requirement 19.3**
        """
        db_path = _make_empty_db()
        try:
            expected = _normalise(plate)
            result = validate_plate(plate, plate_image=None, db_path=db_path)

            assert result["detected_plate"] == expected, (
                f"Input '{plate}': expected detected_plate='{expected}', "
                f"got '{result['detected_plate']}'"
            )
        finally:
            os.unlink(db_path)

    # ------------------------------------------------------------------
    # Spot-checks: known examples
    # ------------------------------------------------------------------

    def test_property_48_known_valid_plate_no_mismatch(self):
        """
        **Property 48: Known valid plate does not get Plate Pattern Mismatch**

        MH34AC3479 is a valid Indian RTO plate with pure chars.
        It should not receive a "Plate Pattern Mismatch" violation.

        **Validates: Requirement 19.1**
        """
        db_path = _make_empty_db()
        try:
            result = validate_plate("MH34AC3479", plate_image=None, db_path=db_path)
            assert result["violation"] != "Plate Pattern Mismatch", (
                f"MH34AC3479 is valid RTO format, got violation='{result['violation']}'"
            )
            assert result["detected_plate"] == "MH34AC3479"
        finally:
            os.unlink(db_path)

    def test_property_48_known_invalid_plate_gets_mismatch(self):
        """
        **Property 48: Known invalid plate gets Plate Pattern Mismatch**

        'MHACMH' has no digits and does not match AA NN AA NNNN.
        Uses only pure letters (M, H, A, C) so no manipulation is detected.
        It should receive a "Plate Pattern Mismatch" violation.

        **Validates: Requirement 19.2**
        """
        db_path = _make_empty_db()
        try:
            result = validate_plate("MHACMH", plate_image=None, db_path=db_path)
            assert result["violation"] == "Plate Pattern Mismatch", (
                f"'MHACMH' is invalid format, got violation='{result['violation']}'"
            )
        finally:
            os.unlink(db_path)

    def test_property_48_spaced_valid_plate_no_mismatch(self):
        """
        **Property 48: Valid plate with spaces is normalised and not mismatched**

        'MH 34 AC 3479' normalises to 'MH34AC3479' which is valid RTO format.

        **Validates: Requirements 19.1, 19.3**
        """
        db_path = _make_empty_db()
        try:
            result = validate_plate("MH 34 AC 3479", plate_image=None, db_path=db_path)
            assert result["detected_plate"] == "MH34AC3479"
            assert result["violation"] != "Plate Pattern Mismatch", (
                f"Spaced valid plate should not get mismatch, "
                f"got violation='{result['violation']}'"
            )
        finally:
            os.unlink(db_path)

    def test_property_48_empty_input_result_fields_present(self):
        """
        **Property 48: Empty input returns result with all required fields**

        **Validates: Requirement 19.3**
        """
        db_path = _make_empty_db()
        try:
            result = validate_plate("", plate_image=None, db_path=db_path)
            assert "detected_plate" in result
            assert "correct_plate" in result
            assert "violation" in result
            assert "confidence_modifier" in result
            assert result["detected_plate"] == ""
            assert 0.0 <= result["confidence_modifier"] <= 1.0
        finally:
            os.unlink(db_path)

    def test_property_48_manipulation_takes_priority_over_mismatch(self):
        """
        **Property 48: Character manipulation violation takes priority over mismatch**

        '0H34' is both invalid format (too short) and contains a substitution
        digit ('0'). The higher-priority "Character Manipulation" violation
        should be returned, not "Plate Pattern Mismatch".

        **Validates: Requirements 19.1, 19.2**
        """
        db_path = _make_empty_db()
        try:
            result = validate_plate("0H34", plate_image=None, db_path=db_path)
            # Manipulation takes priority — must NOT be "Plate Pattern Mismatch"
            assert result["violation"] != "Plate Pattern Mismatch", (
                f"Manipulation should take priority over mismatch, "
                f"got violation='{result['violation']}'"
            )
            assert result["violation"] in ("Character Manipulation", "Tampered Plate"), (
                f"Expected manipulation violation, got '{result['violation']}'"
            )
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
