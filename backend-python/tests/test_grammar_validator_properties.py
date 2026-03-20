"""
Property-Based Tests for Indian RTO Grammar Validation

This module contains property-based tests that verify the grammar validator
correctly validates license plate text against the Indian RTO format standard.

**Validates: Requirements 4.1**

## Test Summary

This test suite implements **Property 8: Grammar Validation Against Indian Format**
from the design document, which states:

"For any plate text input, validate_indian_format() SHALL return is_valid=True
if and only if the plate matches the Indian RTO pattern AA NN AA NNNN
(2 letters, 2 digits, 1–3 letters, 1–4 digits) with a bare length in [6, 12]."

## Test Coverage

1. **Valid plates always accepted**: Hypothesis-generated plates matching the
   AA NN AA NNNN pattern always return is_valid=True.

2. **Pattern-violating plates always rejected**: Plates with wrong character
   types in wrong positions always return is_valid=False.

3. **Length-violating plates always rejected**: Plates whose bare length is
   outside [6, 12] always return is_valid=False.

4. **Normalisation**: Plates with spaces/hyphens that match the pattern after
   stripping are still accepted (is_valid=True).

5. **Empty input**: Empty string always returns is_valid=False.

## Verified Properties

For each call to `validate_indian_format()`, the tests verify:

- **Valid format accepted**: Correctly formatted plates return is_valid=True
- **Invalid format rejected**: Non-conforming plates return is_valid=False
- **Length gate**: Bare length outside [6, 12] is always rejected
- **Normalisation**: Spaces and hyphens are stripped before matching
- **Empty guard**: Empty/blank input is always rejected
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.grammar_validator import validate_indian_format


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def valid_indian_plate(draw):
    """
    Generate a valid Indian RTO plate string.

    Format: AA NN AA NNNN
      - 2 uppercase letters  (state code)
      - 2 digits             (district code)
      - 1–3 uppercase letters (series)
      - 1–4 digits           (registration number)
    """
    state = draw(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=2, max_size=2))
    district = draw(st.text(alphabet="0123456789", min_size=2, max_size=2))
    series = draw(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=3))
    reg = draw(st.text(alphabet="0123456789", min_size=1, max_size=4))
    return state + district + series + reg


@st.composite
def valid_indian_plate_with_separators(draw):
    """
    Generate a valid Indian RTO plate string with spaces or hyphens inserted
    between the four groups (to test normalisation).
    """
    state = draw(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=2, max_size=2))
    district = draw(st.text(alphabet="0123456789", min_size=2, max_size=2))
    series = draw(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=3))
    reg = draw(st.text(alphabet="0123456789", min_size=1, max_size=4))
    sep = draw(st.sampled_from([" ", "-"]))
    return state + sep + district + sep + series + sep + reg


@st.composite
def plate_with_invalid_length(draw):
    """
    Generate a plate whose bare length (spaces/hyphens stripped) is outside [6, 12].
    Either too short (< 6) or too long (> 12).
    """
    too_short = draw(st.booleans())
    if too_short:
        # 0–5 alphanumeric characters
        length = draw(st.integers(min_value=0, max_value=5))
        chars = draw(
            st.text(
                alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                min_size=length,
                max_size=length,
            )
        )
    else:
        # 13–20 alphanumeric characters
        length = draw(st.integers(min_value=13, max_value=20))
        chars = draw(
            st.text(
                alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                min_size=length,
                max_size=length,
            )
        )
    return chars


@st.composite
def plate_with_wrong_pattern(draw):
    """
    Generate a plate whose bare length is in [6, 12] but does NOT match the
    AA NN AA NNNN pattern (wrong character types in wrong positions).

    Strategies used:
      - All digits (no letters)
      - All letters (no digits)
      - Starts with digits instead of letters
      - Letters where digits are expected (positions 2–3)
    """
    strategy = draw(st.integers(min_value=0, max_value=3))

    if strategy == 0:
        # All digits — no letters at all
        length = draw(st.integers(min_value=6, max_value=12))
        plate = draw(st.text(alphabet="0123456789", min_size=length, max_size=length))
    elif strategy == 1:
        # All letters — no digits at all
        length = draw(st.integers(min_value=6, max_value=12))
        plate = draw(
            st.text(
                alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                min_size=length,
                max_size=length,
            )
        )
    elif strategy == 2:
        # Starts with digits (violates the leading 2-letter state code requirement)
        leading_digits = draw(st.text(alphabet="0123456789", min_size=2, max_size=2))
        rest_length = draw(st.integers(min_value=4, max_value=10))
        rest = draw(
            st.text(
                alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                min_size=rest_length,
                max_size=rest_length,
            )
        )
        plate = leading_digits + rest
    else:
        # Letters in district-code positions (positions 2–3 should be digits)
        state = draw(
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=2, max_size=2)
        )
        # Force letters where digits are expected
        bad_district = draw(
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=2, max_size=2)
        )
        rest_length = draw(st.integers(min_value=2, max_value=8))
        rest = draw(
            st.text(
                alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                min_size=rest_length,
                max_size=rest_length,
            )
        )
        plate = state + bad_district + rest

    return plate


# ---------------------------------------------------------------------------
# Property 8: Grammar Validation Against Indian Format
# ---------------------------------------------------------------------------

class TestGrammarValidationProperties:
    """
    Property-based tests for Indian RTO grammar validation.

    Verifies that validate_indian_format() correctly accepts valid plates
    and rejects invalid ones across all possible inputs.

    **Property 8: Grammar Validation Against Indian Format**
    **Validates: Requirements 4.1**
    """

    @given(plate=valid_indian_plate())
    @settings(max_examples=200, deadline=5000)
    def test_property_8_valid_plates_always_accepted(self, plate: str):
        """
        **Property 8: Grammar Validation Against Indian Format**

        For any plate string matching the AA NN AA NNNN pattern with bare
        length in [6, 12], validate_indian_format() SHALL return is_valid=True.

        **Validates: Requirements 4.1**
        """
        result = validate_indian_format(plate)

        assert result["is_valid"] is True, (
            f"Valid Indian plate '{plate}' must be accepted, "
            f"got is_valid=False with reason: '{result['reason']}'"
        )
        assert isinstance(result["plate"], str), "plate field must be a str"
        assert isinstance(result["reason"], str), "reason field must be a str"

    @given(plate=plate_with_wrong_pattern())
    @settings(max_examples=200, deadline=5000)
    def test_property_8_pattern_violating_plates_always_rejected(self, plate: str):
        """
        **Property 8: Grammar Validation Against Indian Format**

        For any plate string that does NOT match the AA NN AA NNNN pattern
        (wrong character types in wrong positions), validate_indian_format()
        SHALL return is_valid=False.

        **Validates: Requirements 4.1**
        """
        result = validate_indian_format(plate)

        assert result["is_valid"] is False, (
            f"Pattern-violating plate '{plate}' must be rejected, "
            f"got is_valid=True"
        )

    @given(plate=plate_with_invalid_length())
    @settings(max_examples=200, deadline=5000)
    def test_property_8_invalid_length_plates_always_rejected(self, plate: str):
        """
        **Property 8: Grammar Validation Against Indian Format**

        For any plate whose bare length (spaces/hyphens stripped) is outside
        [6, 12], validate_indian_format() SHALL return is_valid=False.

        **Validates: Requirements 4.1**
        """
        result = validate_indian_format(plate)

        assert result["is_valid"] is False, (
            f"Plate '{plate}' with bare length {len(plate)} outside [6, 12] "
            f"must be rejected, got is_valid=True"
        )

    @given(plate=valid_indian_plate_with_separators())
    @settings(max_examples=200, deadline=5000)
    def test_property_8_normalisation_spaces_hyphens_accepted(self, plate: str):
        """
        **Property 8: Grammar Validation Against Indian Format — normalisation**

        For any valid Indian plate with spaces or hyphens inserted between
        groups, validate_indian_format() SHALL strip them and return is_valid=True.

        **Validates: Requirements 4.1**
        """
        result = validate_indian_format(plate)

        assert result["is_valid"] is True, (
            f"Valid plate with separators '{plate}' must be accepted after "
            f"normalisation, got is_valid=False with reason: '{result['reason']}'"
        )
        # The returned plate field must be the bare (stripped) form
        bare = plate.replace(" ", "").replace("-", "")
        assert result["plate"] == bare, (
            f"Normalised plate must equal bare form '{bare}', got '{result['plate']}'"
        )

    def test_property_8_empty_input_always_rejected(self):
        """
        **Property 8: Grammar Validation Against Indian Format — empty input**

        An empty string input SHALL always return is_valid=False.

        **Validates: Requirements 4.1**
        """
        for empty in ["", "   ", "\t", "\n"]:
            result = validate_indian_format(empty)
            assert result["is_valid"] is False, (
                f"Empty/blank input {repr(empty)} must return is_valid=False, "
                f"got is_valid=True"
            )

    def test_property_8_known_valid_plates(self):
        """
        **Property 8: Known valid Indian plates are accepted**

        Spot-checks a set of well-known valid Indian RTO plate strings.

        **Validates: Requirements 4.1**
        """
        valid_plates = [
            "MH12AB1234",
            "DL03AF1234",
            "KA01MX9999",
            "TN22Z1",
            "GJ05BQ12",
            "MH 12 AB 1234",   # with spaces
            "DL-03-AF-1234",   # with hyphens
        ]
        for plate in valid_plates:
            result = validate_indian_format(plate)
            assert result["is_valid"] is True, (
                f"Known valid plate '{plate}' must be accepted, "
                f"got is_valid=False with reason: '{result['reason']}'"
            )

    def test_property_8_known_invalid_plates(self):
        """
        **Property 8: Known invalid plates are rejected**

        Spot-checks a set of clearly invalid plate strings.

        **Validates: Requirements 4.1**
        """
        invalid_plates = [
            "",           # empty
            "AB",         # too short
            "1234567890123",  # too long (13 chars)
            "12AB1234",   # starts with digits
            "ABCDE1234",  # no digit district code
            "MH12ABCD12345",  # too long
            "MHAB1234",   # missing district digits
        ]
        for plate in invalid_plates:
            result = validate_indian_format(plate)
            assert result["is_valid"] is False, (
                f"Known invalid plate '{plate}' must be rejected, "
                f"got is_valid=True"
            )

    def test_property_8_result_has_required_keys(self):
        """
        **Property 8: ValidationResult always contains required keys**

        validate_indian_format() SHALL always return a dict with keys:
        is_valid (bool), plate (str), reason (str).

        **Validates: Requirements 4.1**
        """
        for plate in ["MH12AB1234", "INVALID", "", "DL 3C AF 1234"]:
            result = validate_indian_format(plate)
            assert "is_valid" in result, f"Result for '{plate}' missing 'is_valid'"
            assert "plate" in result, f"Result for '{plate}' missing 'plate'"
            assert "reason" in result, f"Result for '{plate}' missing 'reason'"
            assert isinstance(result["is_valid"], bool), (
                f"is_valid must be bool, got {type(result['is_valid'])}"
            )
            assert isinstance(result["plate"], str), (
                f"plate must be str, got {type(result['plate'])}"
            )
            assert isinstance(result["reason"], str), (
                f"reason must be str, got {type(result['reason'])}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
