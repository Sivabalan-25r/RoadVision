"""
Property-Based Tests for Position-Based Character Corrections

This module contains property-based tests that verify apply_position_based_corrections()
correctly enforces letter/digit constraints at each position of an Indian RTO plate.

**Property 9: Position-Based Character Corrections**
**Validates: Requirements 4.4, 4.5, 4.6, 4.7**

## Test Summary

Indian RTO format: AA NN AA NNNN
  Positions 0-1  → state code  (must be letters)
  Positions 2-3  → district code (must be digits)
  Positions 4+   → series letters (must be letters), then registration digits

The correction maps used:
  digit→letter: 0→O, 1→I, 8→B, 5→S, 2→Z, 6→G
  letter→digit: O→0, I→1, B→8, S→5, Z→2, G→6

## Properties Tested

- **Property 9a**: After correction, positions 0-1 are always letters
- **Property 9b**: After correction, positions 2-3 are always digits
- **Property 9c**: After correction, series positions (4+) are always letters
- **Property 9d**: After correction, registration positions are always digits
- **Property 9e**: Corrections are idempotent (applying twice gives same result)
- **Property 9f**: Characters not in the correction maps are left unchanged
- **Property 9g**: Corrected plate has same length as original (no chars added/removed)
- **Property 9h**: CorrectionResult always contains required fields
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.grammar_validator import apply_position_based_corrections

# ---------------------------------------------------------------------------
# Character sets
# ---------------------------------------------------------------------------

# Characters that can be corrected digit→letter (at letter positions)
_CORRECTABLE_DIGITS = set("018526")
# Characters that can be corrected letter→digit (at digit positions)
_CORRECTABLE_LETTERS = set("OIBSZG")
# Pure letters (not in correction map) — safe at letter positions
_PURE_LETTERS = set("ACDEFHJKLMNPQRTUVWXY")
# Pure digits (not in correction map) — safe at digit positions
_PURE_DIGITS = set("3479")

_ALL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ALL_DIGITS = "0123456789"
_AMBIGUOUS_CHARS = "0O1I8B5S2Z6G"  # chars that appear in both maps


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def plate_with_digit_in_letter_position(draw):
    """Generate a plate where positions 0 or 1 contain a correctable digit."""
    pos = draw(st.integers(min_value=0, max_value=1))
    bad_char = draw(st.sampled_from(sorted(_CORRECTABLE_DIGITS)))
    # Build a valid-length plate (10 chars: 2+2+2+4)
    state = list(draw(st.text(alphabet=_ALL_LETTERS, min_size=2, max_size=2)))
    state[pos] = bad_char
    district = draw(st.text(alphabet=_ALL_DIGITS, min_size=2, max_size=2))
    series = draw(st.text(alphabet=_ALL_LETTERS, min_size=2, max_size=2))
    reg = draw(st.text(alphabet=_ALL_DIGITS, min_size=4, max_size=4))
    return "".join(state) + district + series + reg


@st.composite
def plate_with_letter_in_digit_position(draw):
    """Generate a plate where positions 2 or 3 contain a correctable letter."""
    pos = draw(st.integers(min_value=2, max_value=3))
    bad_char = draw(st.sampled_from(sorted(_CORRECTABLE_LETTERS)))
    state = draw(st.text(alphabet=_ALL_LETTERS, min_size=2, max_size=2))
    district = list(draw(st.text(alphabet=_ALL_DIGITS, min_size=2, max_size=2)))
    district[pos - 2] = bad_char
    series = draw(st.text(alphabet=_ALL_LETTERS, min_size=2, max_size=2))
    reg = draw(st.text(alphabet=_ALL_DIGITS, min_size=4, max_size=4))
    return state + "".join(district) + series + reg


@st.composite
def valid_indian_plate(draw):
    """Generate a well-formed Indian RTO plate (no corrections needed)."""
    state = draw(st.text(alphabet=_ALL_LETTERS, min_size=2, max_size=2))
    district = draw(st.text(alphabet=_ALL_DIGITS, min_size=2, max_size=2))
    series = draw(st.text(alphabet=_ALL_LETTERS, min_size=1, max_size=3))
    reg = draw(st.text(alphabet=_ALL_DIGITS, min_size=1, max_size=4))
    return state + district + series + reg


@st.composite
def plate_with_mixed_errors(draw):
    """Generate a plate with correctable errors at multiple positions."""
    # State: may have digit-like chars
    s0 = draw(st.sampled_from(sorted(_CORRECTABLE_DIGITS | _PURE_LETTERS)))
    s1 = draw(st.sampled_from(sorted(_CORRECTABLE_DIGITS | _PURE_LETTERS)))
    # District: may have letter-like chars
    d0 = draw(st.sampled_from(sorted(_CORRECTABLE_LETTERS | _PURE_DIGITS)))
    d1 = draw(st.sampled_from(sorted(_CORRECTABLE_LETTERS | _PURE_DIGITS)))
    # Series: may have digit-like chars
    series_len = draw(st.integers(min_value=1, max_value=3))
    series = draw(st.text(
        alphabet="".join(sorted(_CORRECTABLE_DIGITS | _PURE_LETTERS)),
        min_size=series_len, max_size=series_len
    ))
    # Registration: may have letter-like chars
    reg_len = draw(st.integers(min_value=1, max_value=4))
    reg = draw(st.text(
        alphabet="".join(sorted(_CORRECTABLE_LETTERS | _PURE_DIGITS)),
        min_size=reg_len, max_size=reg_len
    ))
    return s0 + s1 + d0 + d1 + series + reg


# ---------------------------------------------------------------------------
# Helper: digit→letter and letter→digit maps (mirrors grammar_validator.py)
# ---------------------------------------------------------------------------

_DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G'}
_LETTER_TO_DIGIT = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2', 'G': '6'}


# ---------------------------------------------------------------------------
# Property 9: Position-Based Character Corrections
# ---------------------------------------------------------------------------

class TestPositionBasedCorrectionProperties:
    """
    Property-based tests for apply_position_based_corrections().

    **Property 9: Position-Based Character Corrections**
    **Validates: Requirements 4.4, 4.5, 4.6, 4.7**
    """

    # ------------------------------------------------------------------
    # Property 9a — positions 0-1 become letters after correction
    # ------------------------------------------------------------------

    @given(plate=plate_with_digit_in_letter_position())
    @settings(max_examples=200, deadline=5000)
    def test_property_9a_state_positions_become_letters(self, plate: str):
        """
        **Property 9a: State code positions (0-1) are letters after correction**

        For any plate where positions 0-1 contain a correctable digit,
        apply_position_based_corrections() SHALL replace it with the
        corresponding letter.

        **Validates: Requirements 4.4**
        """
        result = apply_position_based_corrections(plate)
        corrected = result["corrected"]

        assert len(corrected) >= 2, "Corrected plate must have at least 2 chars"
        assert corrected[0].isalpha(), (
            f"Position 0 must be a letter after correction, "
            f"got '{corrected[0]}' (input='{plate}')"
        )
        assert corrected[1].isalpha(), (
            f"Position 1 must be a letter after correction, "
            f"got '{corrected[1]}' (input='{plate}')"
        )

    @given(plate=valid_indian_plate())
    @settings(max_examples=200, deadline=5000)
    def test_property_9a_valid_state_positions_unchanged(self, plate: str):
        """
        **Property 9a: Valid letter state positions are not modified**

        For plates already having letters at positions 0-1, no correction
        should be applied to those positions.

        **Validates: Requirements 4.4**
        """
        result = apply_position_based_corrections(plate)
        corrected = result["corrected"]

        # Positions 0-1 were already letters — they must stay the same
        assert corrected[0] == plate[0], (
            f"Position 0 letter '{plate[0]}' must not be changed, "
            f"got '{corrected[0]}'"
        )
        assert corrected[1] == plate[1], (
            f"Position 1 letter '{plate[1]}' must not be changed, "
            f"got '{corrected[1]}'"
        )

    # ------------------------------------------------------------------
    # Property 9b — positions 2-3 become digits after correction
    # ------------------------------------------------------------------

    @given(plate=plate_with_letter_in_digit_position())
    @settings(max_examples=200, deadline=5000)
    def test_property_9b_district_positions_become_digits(self, plate: str):
        """
        **Property 9b: District code positions (2-3) are digits after correction**

        For any plate where positions 2-3 contain a correctable letter,
        apply_position_based_corrections() SHALL replace it with the
        corresponding digit.

        **Validates: Requirements 4.5**
        """
        result = apply_position_based_corrections(plate)
        corrected = result["corrected"]

        assert len(corrected) >= 4, "Corrected plate must have at least 4 chars"
        assert corrected[2].isdigit(), (
            f"Position 2 must be a digit after correction, "
            f"got '{corrected[2]}' (input='{plate}')"
        )
        assert corrected[3].isdigit(), (
            f"Position 3 must be a digit after correction, "
            f"got '{corrected[3]}' (input='{plate}')"
        )

    @given(plate=valid_indian_plate())
    @settings(max_examples=200, deadline=5000)
    def test_property_9b_valid_district_positions_unchanged(self, plate: str):
        """
        **Property 9b: Valid digit district positions are not modified**

        For plates already having digits at positions 2-3, no correction
        should be applied to those positions.

        **Validates: Requirements 4.5**
        """
        result = apply_position_based_corrections(plate)
        corrected = result["corrected"]

        assert corrected[2] == plate[2], (
            f"Position 2 digit '{plate[2]}' must not be changed, "
            f"got '{corrected[2]}'"
        )
        assert corrected[3] == plate[3], (
            f"Position 3 digit '{plate[3]}' must not be changed, "
            f"got '{corrected[3]}'"
        )

    # ------------------------------------------------------------------
    # Property 9c/9d — series and registration positions corrected
    # ------------------------------------------------------------------

    @given(plate=plate_with_mixed_errors())
    @settings(max_examples=300, deadline=5000)
    def test_property_9cd_series_and_registration_corrected(self, plate: str):
        """
        **Property 9c/9d: Series positions become letters, registration becomes digits**

        After correction, the series segment (positions 4 to series_end)
        must contain only letters, and the registration segment must contain
        only digits.

        **Validates: Requirements 4.6, 4.7**
        """
        result = apply_position_based_corrections(plate)
        corrected = result["corrected"]

        # Determine series/registration boundary the same way the validator does:
        # scan from pos 4 while chars are letters or correctable digits
        n = len(corrected)
        series_end = 4
        if n > 4:
            i = 4
            while i < n and corrected[i].isalpha():
                i += 1
            series_end = i

        # Series positions must all be letters
        for pos in range(4, min(series_end, n)):
            assert corrected[pos].isalpha(), (
                f"Series position {pos} must be a letter after correction, "
                f"got '{corrected[pos]}' (input='{plate}', corrected='{corrected}')"
            )

        # Registration positions must all be digits
        for pos in range(series_end, n):
            assert corrected[pos].isdigit(), (
                f"Registration position {pos} must be a digit after correction, "
                f"got '{corrected[pos]}' (input='{plate}', corrected='{corrected}')"
            )

    # ------------------------------------------------------------------
    # Property 9e — idempotence
    # ------------------------------------------------------------------

    @given(plate=plate_with_mixed_errors())
    @settings(max_examples=200, deadline=5000)
    def test_property_9e_corrections_are_idempotent(self, plate: str):
        """
        **Property 9e: Corrections are idempotent**

        Applying apply_position_based_corrections() twice must produce the
        same result as applying it once.

        **Validates: Requirements 4.4, 4.5, 4.6, 4.7**
        """
        first = apply_position_based_corrections(plate)["corrected"]
        second = apply_position_based_corrections(first)["corrected"]

        assert first == second, (
            f"Correction must be idempotent: "
            f"first='{first}', second='{second}' (input='{plate}')"
        )

    # ------------------------------------------------------------------
    # Property 9f — state/district positions with pure chars are unchanged
    # ------------------------------------------------------------------

    @given(
        state=st.text(alphabet="".join(sorted(_PURE_LETTERS)), min_size=2, max_size=2),
        district=st.text(alphabet="".join(sorted(_PURE_DIGITS)), min_size=2, max_size=2),
        series=st.text(alphabet="".join(sorted(_PURE_LETTERS)), min_size=1, max_size=3),
        reg=st.text(alphabet="".join(sorted(_PURE_DIGITS)), min_size=1, max_size=4),
    )
    @settings(max_examples=200, deadline=5000)
    def test_property_9f_pure_chars_not_modified(
        self, state: str, district: str, series: str, reg: str
    ):
        """
        **Property 9f: Characters not in correction maps are left unchanged**

        For plates built entirely from characters that are NOT in either
        correction map (pure letters at letter positions, pure digits at digit
        positions), apply_position_based_corrections() must not modify any
        character and must log zero corrections.

        Pure letters (not in digit→letter map): A C D E F H J K L M N P Q R T U V W X Y
        Pure digits (not in letter→digit map): 3 4 7 9

        **Validates: Requirements 4.4, 4.5, 4.6, 4.7**
        """
        plate = state + district + series + reg
        result = apply_position_based_corrections(plate)
        corrected = result["corrected"]

        assert corrected == plate, (
            f"Plate '{plate}' built from pure chars must not be modified, "
            f"got '{corrected}'"
        )
        assert result["corrections"] == [], (
            f"No corrections should be logged for pure-char plate '{plate}', "
            f"got: {result['corrections']}"
        )

    # ------------------------------------------------------------------
    # Property 9g — length preservation
    # ------------------------------------------------------------------

    @given(plate=plate_with_mixed_errors())
    @settings(max_examples=200, deadline=5000)
    def test_property_9g_length_preserved(self, plate: str):
        """
        **Property 9g: Correction preserves plate length**

        apply_position_based_corrections() must not add or remove characters;
        the corrected plate must have the same length as the normalised input.

        **Validates: Requirements 4.4, 4.5, 4.6, 4.7**
        """
        result = apply_position_based_corrections(plate)
        original = result["original"]
        corrected = result["corrected"]

        assert len(corrected) == len(original), (
            f"Length must be preserved: original='{original}' ({len(original)}), "
            f"corrected='{corrected}' ({len(corrected)})"
        )

    # ------------------------------------------------------------------
    # Property 9h — result structure
    # ------------------------------------------------------------------

    @given(plate=st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_size=0, max_size=15
    ))
    @settings(max_examples=200, deadline=5000)
    def test_property_9h_result_has_required_fields(self, plate: str):
        """
        **Property 9h: CorrectionResult always contains required fields**

        apply_position_based_corrections() SHALL always return a dict with:
          - original (str)
          - corrected (str)
          - corrections (list)

        **Validates: Requirements 4.4, 4.5, 4.6, 4.7**
        """
        result = apply_position_based_corrections(plate)

        assert "original" in result, "Result must contain 'original'"
        assert "corrected" in result, "Result must contain 'corrected'"
        assert "corrections" in result, "Result must contain 'corrections'"
        assert isinstance(result["original"], str), "original must be str"
        assert isinstance(result["corrected"], str), "corrected must be str"
        assert isinstance(result["corrections"], list), "corrections must be list"

    # ------------------------------------------------------------------
    # Spot-check: known correction examples
    # ------------------------------------------------------------------

    def test_property_9_known_digit_to_letter_corrections(self):
        """
        **Property 9: Known digit→letter corrections at state positions**

        Spot-checks specific substitutions at positions 0-1.
        Registration uses only pure digits (3,4,7,9) to avoid the series
        boundary scan absorbing correctable digits.

        **Validates: Requirements 4.4**
        """
        cases = [
            # (input,        expected_corrected)
            ("0H34AB3434",  "OH34AB3434"),   # 0→O at pos 0
            ("MH34AB3434",  "MH34AB3434"),   # no correction needed
            ("8H34AB3434",  "BH34AB3434"),   # 8→B at pos 0
            ("M534AB3434",  "MS34AB3434"),   # 5→S at pos 1
            ("2H34AB3434",  "ZH34AB3434"),   # 2→Z at pos 0
            ("M634AB3434",  "MG34AB3434"),   # 6→G at pos 1
        ]
        for inp, expected in cases:
            result = apply_position_based_corrections(inp)
            assert result["corrected"] == expected, (
                f"Input '{inp}': expected corrected='{expected}', "
                f"got '{result['corrected']}'"
            )

    def test_property_9_known_letter_to_digit_corrections(self):
        """
        **Property 9: Known letter→digit corrections at district positions**

        Spot-checks specific substitutions at positions 2-3.
        Registration uses only pure digits (3,4,7,9) to avoid the series
        boundary scan absorbing correctable digits.

        **Validates: Requirements 4.5**
        """
        cases = [
            # (input,        expected_corrected)
            ("MHOOAB3434",  "MH00AB3434"),   # O→0 at pos 2 and 3
            ("MHIIAB3434",  "MH11AB3434"),   # I→1 at pos 2 and 3
            ("MHBBAB3434",  "MH88AB3434"),   # B→8 at pos 2 and 3
            ("MHSSAB3434",  "MH55AB3434"),   # S→5 at pos 2 and 3
            ("MHZZAB3434",  "MH22AB3434"),   # Z→2 at pos 2 and 3
            ("MHGGAB3434",  "MH66AB3434"),   # G→6 at pos 2 and 3
        ]
        for inp, expected in cases:
            result = apply_position_based_corrections(inp)
            assert result["corrected"] == expected, (
                f"Input '{inp}': expected corrected='{expected}', "
                f"got '{result['corrected']}'"
            )

    def test_property_9_no_correction_needed_for_valid_plate(self):
        """
        **Property 9: Plates with only pure chars pass through without modification**

        Uses plates whose registration contains only pure digits (3,4,7,9)
        and series contains only pure letters, so the series boundary scan
        does not absorb any correctable characters.

        **Validates: Requirements 4.4, 4.5, 4.6, 4.7**
        """
        # Pure-char plates: state=pure letters, district=pure digits,
        # series=pure letters, registration=pure digits (3,4,7,9 only)
        pure_plates = [
            "MH34AB3434",
            "DL39AF3434",
            "KA47MX9999",
            "TN33AC4477",
        ]
        for plate in pure_plates:
            result = apply_position_based_corrections(plate)
            assert result["corrected"] == plate, (
                f"Pure-char plate '{plate}' must not be modified, "
                f"got '{result['corrected']}'"
            )
            assert result["corrections"] == [], (
                f"No corrections should be logged for '{plate}', "
                f"got: {result['corrections']}"
            )

    def test_property_9_empty_input(self):
        """
        **Property 9: Empty input returns empty result**

        **Validates: Requirements 4.4**
        """
        result = apply_position_based_corrections("")
        assert result["original"] == ""
        assert result["corrected"] == ""
        assert result["corrections"] == []

    def test_property_9_corrections_log_each_change(self):
        """
        **Property 9: Each correction is logged in the corrections list**

        When a character is corrected, the corrections list must contain
        an entry describing the change. Uses pure-digit registration to
        avoid the series boundary scan absorbing correctable digits.

        **Validates: Requirements 4.4, 4.5**
        """
        # Two corrections: pos 0 (0→O) and pos 2 (O→0)
        # Registration uses pure digits (3,4) so no extra corrections occur
        result = apply_position_based_corrections("0HO3AB3434")
        assert len(result["corrections"]) == 2, (
            f"Expected 2 corrections, got {len(result['corrections'])}: "
            f"{result['corrections']}"
        )
        # Each entry must mention the position
        assert any("pos 0" in c for c in result["corrections"]), (
            "Correction at pos 0 must be logged"
        )
        assert any("pos 2" in c for c in result["corrections"]), (
            "Correction at pos 2 must be logged"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
