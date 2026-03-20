"""
Property-Based Tests for Character Manipulation Detection

This module contains property-based tests that verify detect_character_manipulation()
correctly identifies suspicious character substitutions in license plate strings.

**Property 42: Character Manipulation Detection**
**Validates: Requirements 16.1, 16.2**

## Test Summary

Requirement 16.1: The system shall detect common character substitutions used
to manipulate license plates (e.g., 0→O, 1→I, 8→B, 5→S, 2→Z, 6→G).

Requirement 16.2: The system shall flag plates with detected character
manipulation as a violation (is_manipulation=True).

## Properties Tested

- **Property 42a**: Any plate containing a known substitution character
  (0, 1, 8, 5, 2, 6 in any position, or O, I, B, S, Z, G in any position)
  is detected as manipulated (is_manipulation=True).

- **Property 42b**: Plates with detected manipulation are flagged as violations
  (is_manipulation=True and substitutions list is non-empty).

- **Property 42c**: Plates containing only pure letters and pure digits
  (no ambiguous chars) are NOT flagged as manipulated.

- **Property 42d**: ManipulationResult always contains required fields with
  correct types.

- **Property 42e**: The substitutions list length matches the number of
  ambiguous characters in the plate.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.grammar_validator import (
    detect_character_manipulation,
    _DIGIT_TO_LETTER,
    _LETTER_TO_DIGIT,
)

# ---------------------------------------------------------------------------
# Character sets (mirrors grammar_validator.py)
# ---------------------------------------------------------------------------

# Digits that look like letters — flagged when found anywhere in the plate
_SUBSTITUTION_DIGITS = set("018526")   # 0→O, 1→I, 8→B, 5→S, 2→Z, 6→G

# Letters that look like digits — flagged when found anywhere in the plate
_SUBSTITUTION_LETTERS = set("OIBSZG")  # O→0, I→1, B→8, S→5, Z→2, G→6

# All ambiguous characters (union of both sets)
_AMBIGUOUS_CHARS = _SUBSTITUTION_DIGITS | _SUBSTITUTION_LETTERS

# Pure letters: uppercase letters NOT in the substitution-letter set
_PURE_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - _SUBSTITUTION_LETTERS

# Pure digits: digits NOT in the substitution-digit set
_PURE_DIGITS = set("0123456789") - _SUBSTITUTION_DIGITS


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def plate_with_substitution_digit(draw):
    """
    Generate a plate that contains at least one substitution digit
    (0, 1, 8, 5, 2, or 6) somewhere in the string.
    """
    # Pick a substitution digit to inject
    bad_char = draw(st.sampled_from(sorted(_SUBSTITUTION_DIGITS)))
    # Build a base plate from pure chars (no ambiguous chars)
    length = draw(st.integers(min_value=5, max_value=11))
    base = draw(st.text(
        alphabet="".join(sorted(_PURE_LETTERS | _PURE_DIGITS)),
        min_size=length,
        max_size=length,
    ))
    # Inject the substitution digit at a random position
    pos = draw(st.integers(min_value=0, max_value=length))
    plate = base[:pos] + bad_char + base[pos:]
    return plate


@st.composite
def plate_with_substitution_letter(draw):
    """
    Generate a plate that contains at least one substitution letter
    (O, I, B, S, Z, or G) somewhere in the string.
    """
    bad_char = draw(st.sampled_from(sorted(_SUBSTITUTION_LETTERS)))
    length = draw(st.integers(min_value=5, max_value=11))
    base = draw(st.text(
        alphabet="".join(sorted(_PURE_LETTERS | _PURE_DIGITS)),
        min_size=length,
        max_size=length,
    ))
    pos = draw(st.integers(min_value=0, max_value=length))
    plate = base[:pos] + bad_char + base[pos:]
    return plate


@st.composite
def plate_with_only_pure_chars(draw):
    """
    Generate a plate built entirely from pure letters and pure digits
    (no ambiguous characters). These plates must NOT be flagged as manipulated.
    """
    length = draw(st.integers(min_value=6, max_value=12))
    plate = draw(st.text(
        alphabet="".join(sorted(_PURE_LETTERS | _PURE_DIGITS)),
        min_size=length,
        max_size=length,
    ))
    return plate


@st.composite
def plate_with_multiple_substitutions(draw):
    """
    Generate a plate with 2–4 ambiguous characters injected at distinct positions.
    """
    length = draw(st.integers(min_value=6, max_value=10))
    base = list(draw(st.text(
        alphabet="".join(sorted(_PURE_LETTERS | _PURE_DIGITS)),
        min_size=length,
        max_size=length,
    )))
    # Inject 2–4 ambiguous chars at distinct positions
    num_subs = draw(st.integers(min_value=2, max_value=min(4, length)))
    positions = draw(st.lists(
        st.integers(min_value=0, max_value=length - 1),
        min_size=num_subs,
        max_size=num_subs,
        unique=True,
    ))
    for pos in positions:
        bad_char = draw(st.sampled_from(sorted(_AMBIGUOUS_CHARS)))
        base[pos] = bad_char
    return "".join(base)


# ---------------------------------------------------------------------------
# Property 42: Character Manipulation Detection
# ---------------------------------------------------------------------------

class TestCharacterManipulationDetectionProperties:
    """
    Property-based tests for detect_character_manipulation().

    **Property 42: Character Manipulation Detection**
    **Validates: Requirements 16.1, 16.2**
    """

    # ------------------------------------------------------------------
    # Property 42a — substitution digits are always detected
    # ------------------------------------------------------------------

    @given(plate=plate_with_substitution_digit())
    @settings(max_examples=200, deadline=5000)
    def test_property_42a_substitution_digit_detected(self, plate: str):
        """
        **Property 42a: Plates with substitution digits are detected as manipulated**

        For any plate containing at least one digit from {0, 1, 8, 5, 2, 6},
        detect_character_manipulation() SHALL return is_manipulation=True.

        **Validates: Requirements 16.1, 16.2**
        """
        result = detect_character_manipulation(plate)

        assert result["is_manipulation"] is True, (
            f"Plate '{plate}' contains a substitution digit and must be "
            f"flagged as manipulated, got is_manipulation=False"
        )
        assert len(result["substitutions"]) > 0, (
            f"Plate '{plate}' must have at least one substitution entry, "
            f"got empty substitutions list"
        )

    # ------------------------------------------------------------------
    # Property 42a (cont.) — substitution letters are always detected
    # ------------------------------------------------------------------

    @given(plate=plate_with_substitution_letter())
    @settings(max_examples=200, deadline=5000)
    def test_property_42a_substitution_letter_detected(self, plate: str):
        """
        **Property 42a: Plates with substitution letters are detected as manipulated**

        For any plate containing at least one letter from {O, I, B, S, Z, G},
        detect_character_manipulation() SHALL return is_manipulation=True.

        **Validates: Requirements 16.1, 16.2**
        """
        result = detect_character_manipulation(plate)

        assert result["is_manipulation"] is True, (
            f"Plate '{plate}' contains a substitution letter and must be "
            f"flagged as manipulated, got is_manipulation=False"
        )
        assert len(result["substitutions"]) > 0, (
            f"Plate '{plate}' must have at least one substitution entry, "
            f"got empty substitutions list"
        )

    # ------------------------------------------------------------------
    # Property 42b — is_manipulation and substitutions are consistent
    # ------------------------------------------------------------------

    @given(plate=plate_with_multiple_substitutions())
    @settings(max_examples=200, deadline=5000)
    def test_property_42b_manipulation_flag_consistent_with_substitutions(
        self, plate: str
    ):
        """
        **Property 42b: is_manipulation is True iff substitutions list is non-empty**

        For any plate with multiple injected ambiguous characters,
        is_manipulation must be True and the substitutions list must be
        non-empty (violation flagged).

        **Validates: Requirements 16.2**
        """
        result = detect_character_manipulation(plate)

        assert result["is_manipulation"] is True, (
            f"Plate '{plate}' with multiple substitutions must be flagged, "
            f"got is_manipulation=False"
        )
        assert len(result["substitutions"]) >= 2, (
            f"Plate '{plate}' must have ≥2 substitution entries, "
            f"got {len(result['substitutions'])}: {result['substitutions']}"
        )

    # ------------------------------------------------------------------
    # Property 42c — pure-char plates are NOT flagged
    # ------------------------------------------------------------------

    @given(plate=plate_with_only_pure_chars())
    @settings(max_examples=200, deadline=5000)
    def test_property_42c_pure_plates_not_flagged(self, plate: str):
        """
        **Property 42c: Plates with only pure chars are not flagged as manipulated**

        For any plate built entirely from characters that are NOT in either
        substitution map, detect_character_manipulation() SHALL return
        is_manipulation=False and an empty substitutions list.

        **Validates: Requirements 16.1**
        """
        result = detect_character_manipulation(plate)

        assert result["is_manipulation"] is False, (
            f"Pure-char plate '{plate}' must NOT be flagged as manipulated, "
            f"got is_manipulation=True with substitutions: {result['substitutions']}"
        )
        assert result["substitutions"] == [], (
            f"Pure-char plate '{plate}' must have empty substitutions list, "
            f"got: {result['substitutions']}"
        )

    # ------------------------------------------------------------------
    # Property 42d — result always has required fields
    # ------------------------------------------------------------------

    @given(plate=st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_size=0,
        max_size=15,
    ))
    @settings(max_examples=200, deadline=5000)
    def test_property_42d_result_has_required_fields(self, plate: str):
        """
        **Property 42d: ManipulationResult always contains required fields**

        detect_character_manipulation() SHALL always return a dict with:
          - original (str)
          - corrected (str)
          - is_manipulation (bool)
          - substitutions (list)

        **Validates: Requirements 16.1, 16.2**
        """
        result = detect_character_manipulation(plate)

        assert "original" in result, "Result must contain 'original'"
        assert "corrected" in result, "Result must contain 'corrected'"
        assert "is_manipulation" in result, "Result must contain 'is_manipulation'"
        assert "substitutions" in result, "Result must contain 'substitutions'"

        assert isinstance(result["original"], str), "original must be str"
        assert isinstance(result["corrected"], str), "corrected must be str"
        assert isinstance(result["is_manipulation"], bool), "is_manipulation must be bool"
        assert isinstance(result["substitutions"], list), "substitutions must be list"

    # ------------------------------------------------------------------
    # Property 42e — substitutions count matches ambiguous chars in plate
    # ------------------------------------------------------------------

    @given(plate=st.text(
        alphabet="".join(sorted(_PURE_LETTERS | _PURE_DIGITS | _AMBIGUOUS_CHARS)),
        min_size=1,
        max_size=12,
    ))
    @settings(max_examples=300, deadline=5000)
    def test_property_42e_substitutions_count_matches_ambiguous_chars(
        self, plate: str
    ):
        """
        **Property 42e: Substitutions list length equals number of ambiguous chars**

        The number of entries in the substitutions list must equal the count
        of ambiguous characters (from _SUBSTITUTION_DIGITS ∪ _SUBSTITUTION_LETTERS)
        in the normalised plate.

        **Validates: Requirements 16.1**
        """
        result = detect_character_manipulation(plate)
        normalised = plate.upper()

        expected_count = sum(1 for ch in normalised if ch in _AMBIGUOUS_CHARS)

        assert len(result["substitutions"]) == expected_count, (
            f"Plate '{plate}' (normalised='{normalised}'): expected "
            f"{expected_count} substitution entries, "
            f"got {len(result['substitutions'])}: {result['substitutions']}"
        )

    # ------------------------------------------------------------------
    # Spot-checks: known examples from the docstring
    # ------------------------------------------------------------------

    def test_property_42_known_clean_plate_not_flagged(self):
        """
        **Property 42: Known clean plate is not flagged**

        MH34AC3479 uses only pure letters (M, H, A, C) and pure digits
        (3, 4, 7, 9) — none of which are in the substitution maps.
        Must not be flagged.

        **Validates: Requirements 16.1**
        """
        result = detect_character_manipulation("MH34AC3479")
        assert result["is_manipulation"] is False
        assert result["substitutions"] == []
        assert result["original"] == "MH34AC3479"

    def test_property_42_known_manipulated_plate_flagged(self):
        """
        **Property 42: Known manipulated plate is flagged**

        0H34AC3479 has '0' at position 0 (digit that looks like 'O').
        All other characters are pure, so exactly 1 substitution is detected.

        **Validates: Requirements 16.1, 16.2**
        """
        result = detect_character_manipulation("0H34AC3479")
        assert result["is_manipulation"] is True
        assert len(result["substitutions"]) == 1
        assert "pos 0" in result["substitutions"][0]
        assert result["original"] == "0H34AC3479"

    def test_property_42_all_substitution_digits_detected(self):
        """
        **Property 42: Each substitution digit is individually detected**

        Tests each of the six substitution digits (0, 1, 8, 5, 2, 6) in
        isolation to confirm all are detected.

        **Validates: Requirements 16.1**
        """
        # Each plate has one substitution digit at position 0
        # Use pure chars elsewhere to avoid extra detections
        cases = [
            ("0H34AC3434", "0"),   # 0 → O
            ("1H34AC3434", "1"),   # 1 → I
            ("8H34AC3434", "8"),   # 8 → B
            ("5H34AC3434", "5"),   # 5 → S
            ("2H34AC3434", "2"),   # 2 → Z
            ("6H34AC3434", "6"),   # 6 → G
        ]
        for plate, char in cases:
            result = detect_character_manipulation(plate)
            assert result["is_manipulation"] is True, (
                f"Plate '{plate}' with substitution digit '{char}' must be flagged"
            )
            assert len(result["substitutions"]) == 1, (
                f"Plate '{plate}' must have exactly 1 substitution, "
                f"got {len(result['substitutions'])}"
            )

    def test_property_42_all_substitution_letters_detected(self):
        """
        **Property 42: Each substitution letter is individually detected**

        Tests each of the six substitution letters (O, I, B, S, Z, G) in
        isolation to confirm all are detected.

        **Validates: Requirements 16.1**
        """
        # Each plate has one substitution letter at position 0
        # Use pure chars elsewhere to avoid extra detections
        cases = [
            ("OH34AC3434", "O"),   # O → 0
            ("IH34AC3434", "I"),   # I → 1
            ("BH34AC3434", "B"),   # B → 8
            ("SH34AC3434", "S"),   # S → 5
            ("ZH34AC3434", "Z"),   # Z → 2
            ("GH34AC3434", "G"),   # G → 6
        ]
        for plate, char in cases:
            result = detect_character_manipulation(plate)
            assert result["is_manipulation"] is True, (
                f"Plate '{plate}' with substitution letter '{char}' must be flagged"
            )
            assert len(result["substitutions"]) == 1, (
                f"Plate '{plate}' must have exactly 1 substitution, "
                f"got {len(result['substitutions'])}"
            )

    def test_property_42_empty_input_not_flagged(self):
        """
        **Property 42: Empty input returns is_manipulation=False**

        **Validates: Requirements 16.1**
        """
        result = detect_character_manipulation("")
        assert result["is_manipulation"] is False
        assert result["substitutions"] == []
        assert result["original"] == ""
        assert result["corrected"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ===========================================================================
# Property 43: Character Manipulation Correction
# Validates: Requirements 16.3, 16.4
# ===========================================================================

class TestCharacterManipulationCorrectionProperties:
    """
    Property-based tests for the ``corrected`` field returned by
    detect_character_manipulation().

    **Property 43: Character Manipulation Correction**
    **Validates: Requirements 16.3, 16.4**

    Requirement 16.3: When character manipulation is detected, the system
    shall produce a corrected plate string with ambiguous characters resolved
    using position-based rules.

    Requirement 16.4: The corrected plate shall contain no ambiguous
    characters in positions where the type (letter vs digit) is unambiguous
    according to the Indian RTO format.
    """

    # ------------------------------------------------------------------
    # Property 43a — corrected plate has no ambiguous chars in typed positions
    # ------------------------------------------------------------------

    @given(plate=st.text(
        alphabet="".join(sorted(_PURE_LETTERS | _PURE_DIGITS | _AMBIGUOUS_CHARS)),
        min_size=6,
        max_size=12,
    ))
    @settings(max_examples=300, deadline=5000)
    def test_property_43a_corrected_has_no_ambiguous_chars_in_typed_positions(
        self, plate: str
    ):
        """
        **Property 43a: Corrected plate resolves ambiguous chars in typed positions**

        For any plate of length 6–12, the corrected plate returned by
        detect_character_manipulation() must satisfy:
          - Positions 0-1 (state code): only pure letters (no substitution digits)
          - Positions 2-3 (district code): only pure digits (no substitution letters)

        **Validates: Requirements 16.3, 16.4**
        """
        result = detect_character_manipulation(plate)
        corrected = result["corrected"]

        # Positions 0-1 must be letters (no substitution digits)
        for pos in range(min(2, len(corrected))):
            ch = corrected[pos]
            assert ch not in _SUBSTITUTION_DIGITS, (
                f"Plate '{plate}' → corrected '{corrected}': "
                f"pos {pos} is '{ch}' (substitution digit) but must be a letter"
            )

        # Positions 2-3 must be digits (no substitution letters)
        for pos in range(2, min(4, len(corrected))):
            ch = corrected[pos]
            assert ch not in _SUBSTITUTION_LETTERS, (
                f"Plate '{plate}' → corrected '{corrected}': "
                f"pos {pos} is '{ch}' (substitution letter) but must be a digit"
            )

    # ------------------------------------------------------------------
    # Property 43b — original is preserved unchanged in result
    # ------------------------------------------------------------------

    @given(plate=st.text(
        alphabet="".join(sorted(_PURE_LETTERS | _PURE_DIGITS | _AMBIGUOUS_CHARS)),
        min_size=1,
        max_size=12,
    ))
    @settings(max_examples=300, deadline=5000)
    def test_property_43b_original_field_is_normalised_input(self, plate: str):
        """
        **Property 43b: original field equals normalised (stripped, uppercased) input**

        detect_character_manipulation() must store the normalised bare plate
        (spaces/hyphens removed, uppercased) in the ``original`` field without
        applying any corrections to it.

        **Validates: Requirements 16.3**
        """
        import re as _re
        expected_original = _re.sub(r'[\s\-]+', '', plate.strip().upper())

        result = detect_character_manipulation(plate)

        assert result["original"] == expected_original, (
            f"Input '{plate}': expected original='{expected_original}', "
            f"got '{result['original']}'"
        )

    # ------------------------------------------------------------------
    # Property 43c — corrected length equals original length
    # ------------------------------------------------------------------

    @given(plate=st.text(
        alphabet="".join(sorted(_PURE_LETTERS | _PURE_DIGITS | _AMBIGUOUS_CHARS)),
        min_size=1,
        max_size=12,
    ))
    @settings(max_examples=300, deadline=5000)
    def test_property_43c_corrected_same_length_as_original(self, plate: str):
        """
        **Property 43c: Corrected plate has the same length as the original**

        Correction only substitutes characters; it must never add or remove
        characters, so len(corrected) == len(original).

        **Validates: Requirements 16.3**
        """
        result = detect_character_manipulation(plate)

        assert len(result["corrected"]) == len(result["original"]), (
            f"Plate '{plate}': original length {len(result['original'])} "
            f"!= corrected length {len(result['corrected'])}"
        )

    # ------------------------------------------------------------------
    # Property 43d — pure plates are returned unchanged
    # ------------------------------------------------------------------

    @given(plate=plate_with_only_pure_chars())
    @settings(max_examples=200, deadline=5000)
    def test_property_43d_pure_plates_corrected_equals_original(self, plate: str):
        """
        **Property 43d: Pure plates are returned unchanged in corrected field**

        When a plate contains no ambiguous characters, the corrected field
        must equal the original field (no unnecessary changes).

        **Validates: Requirements 16.3, 16.4**
        """
        result = detect_character_manipulation(plate)

        assert result["corrected"] == result["original"], (
            f"Pure plate '{plate}': corrected '{result['corrected']}' "
            f"!= original '{result['original']}' (no changes expected)"
        )

    # ------------------------------------------------------------------
    # Property 43e — corrected plate contains only alphanumeric characters
    # ------------------------------------------------------------------

    @given(plate=st.text(
        alphabet="".join(sorted(_PURE_LETTERS | _PURE_DIGITS | _AMBIGUOUS_CHARS)),
        min_size=1,
        max_size=12,
    ))
    @settings(max_examples=300, deadline=5000)
    def test_property_43e_corrected_is_alphanumeric(self, plate: str):
        """
        **Property 43e: Corrected plate contains only alphanumeric characters**

        The correction process must not introduce spaces, hyphens, or any
        non-alphanumeric characters into the corrected plate.

        **Validates: Requirements 16.3**
        """
        result = detect_character_manipulation(plate)
        corrected = result["corrected"]

        assert corrected.isalnum() or corrected == "", (
            f"Plate '{plate}' → corrected '{corrected}' contains non-alphanumeric chars"
        )

    # ------------------------------------------------------------------
    # Property 43f — substitution digit at pos 0 is corrected to a letter
    # ------------------------------------------------------------------

    @given(
        sub_digit=st.sampled_from(sorted(_SUBSTITUTION_DIGITS)),
        suffix=st.text(
            alphabet="".join(sorted(_PURE_LETTERS | _PURE_DIGITS)),
            min_size=5,
            max_size=11,
        ),
    )
    @settings(max_examples=200, deadline=5000)
    def test_property_43f_substitution_digit_at_pos0_corrected_to_letter(
        self, sub_digit: str, suffix: str
    ):
        """
        **Property 43f: Substitution digit at position 0 is corrected to a letter**

        Position 0 is the first character of the state code and must be a
        letter.  Any substitution digit placed there must be replaced with
        its letter equivalent in the corrected plate.

        **Validates: Requirements 16.3, 16.4**
        """
        plate = sub_digit + suffix
        result = detect_character_manipulation(plate)
        corrected = result["corrected"]

        assert len(corrected) > 0, f"Corrected plate must not be empty for input '{plate}'"
        assert corrected[0].isalpha(), (
            f"Plate '{plate}': corrected[0]='{corrected[0]}' must be a letter "
            f"(substitution digit '{sub_digit}' at pos 0 should be corrected)"
        )
        assert corrected[0] == _DIGIT_TO_LETTER[sub_digit], (
            f"Plate '{plate}': corrected[0]='{corrected[0]}' should be "
            f"'{_DIGIT_TO_LETTER[sub_digit]}' (correction of '{sub_digit}')"
        )

    # ------------------------------------------------------------------
    # Property 43g — substitution letter at pos 2 is corrected to a digit
    # ------------------------------------------------------------------

    @given(
        prefix=st.text(
            alphabet="".join(sorted(_PURE_LETTERS)),
            min_size=2,
            max_size=2,
        ),
        sub_letter=st.sampled_from(sorted(_SUBSTITUTION_LETTERS)),
        suffix=st.text(
            alphabet="".join(sorted(_PURE_LETTERS | _PURE_DIGITS)),
            min_size=4,
            max_size=9,
        ),
    )
    @settings(max_examples=200, deadline=5000)
    def test_property_43g_substitution_letter_at_pos2_corrected_to_digit(
        self, prefix: str, sub_letter: str, suffix: str
    ):
        """
        **Property 43g: Substitution letter at position 2 is corrected to a digit**

        Position 2 is the first character of the district code and must be a
        digit.  Any substitution letter placed there must be replaced with
        its digit equivalent in the corrected plate.

        **Validates: Requirements 16.3, 16.4**
        """
        plate = prefix + sub_letter + suffix
        result = detect_character_manipulation(plate)
        corrected = result["corrected"]

        assert len(corrected) >= 3, (
            f"Corrected plate '{corrected}' too short for input '{plate}'"
        )
        assert corrected[2].isdigit(), (
            f"Plate '{plate}': corrected[2]='{corrected[2]}' must be a digit "
            f"(substitution letter '{sub_letter}' at pos 2 should be corrected)"
        )
        assert corrected[2] == _LETTER_TO_DIGIT[sub_letter], (
            f"Plate '{plate}': corrected[2]='{corrected[2]}' should be "
            f"'{_LETTER_TO_DIGIT[sub_letter]}' (correction of '{sub_letter}')"
        )

    # ------------------------------------------------------------------
    # Spot-checks: known correction examples
    # ------------------------------------------------------------------

    def test_property_43_known_correction_digit_to_letter(self):
        """
        **Property 43: '0H34AC3479' corrected to 'OH34AC3479'**

        '0' at position 0 (state code) must be corrected to 'O'.
        Uses only pure digits (3, 4, 7, 9) in the registration to avoid
        additional substitution corrections.

        **Validates: Requirements 16.3, 16.4**
        """
        result = detect_character_manipulation("0H34AC3479")
        assert result["original"] == "0H34AC3479"
        assert result["corrected"] == "OH34AC3479"
        assert result["is_manipulation"] is True

    def test_property_43_known_correction_letter_to_digit(self):
        """
        **Property 43: 'MHO4AC3479' corrected to 'MH04AC3479'**

        'O' at position 2 (district code) must be corrected to '0'.
        Uses only pure digits (3, 4, 7, 9) in the registration.

        **Validates: Requirements 16.3, 16.4**
        """
        result = detect_character_manipulation("MHO4AC3479")
        assert result["original"] == "MHO4AC3479"
        assert result["corrected"] == "MH04AC3479"
        assert result["is_manipulation"] is True

    def test_property_43_multiple_corrections(self):
        """
        **Property 43: Multiple ambiguous chars are all corrected**

        '0HO4AC3479': '0' at pos 0 → 'O', 'O' at pos 2 → '0'.
        Uses only pure digits (3, 4, 7, 9) in the registration.

        **Validates: Requirements 16.3, 16.4**
        """
        result = detect_character_manipulation("0HO4AC3479")
        assert result["original"] == "0HO4AC3479"
        assert result["corrected"] == "OH04AC3479"
        assert result["is_manipulation"] is True
        assert len(result["substitutions"]) == 2

    def test_property_43_clean_plate_unchanged(self):
        """
        **Property 43: Clean plate is returned unchanged**

        'MH34AC3479' uses only pure letters (M, H, A, C) and pure digits
        (3, 4, 7, 9); corrected must equal original.

        **Validates: Requirements 16.3**
        """
        result = detect_character_manipulation("MH34AC3479")
        assert result["original"] == "MH34AC3479"
        assert result["corrected"] == "MH34AC3479"
        assert result["is_manipulation"] is False

    def test_property_43_empty_input_corrected_is_empty(self):
        """
        **Property 43: Empty input returns empty corrected field**

        **Validates: Requirements 16.3**
        """
        result = detect_character_manipulation("")
        assert result["original"] == ""
        assert result["corrected"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
