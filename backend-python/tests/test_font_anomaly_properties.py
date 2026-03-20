"""
Property-Based Tests for Font Anomaly Detection

This module contains property-based tests that verify detect_font_anomalies()
returns a well-formed FontAnomalyResult with correct output types and semantics.

**Property 44: Font Anomaly Detection Output**
**Validates: Requirements 17.3, 17.4, 17.5**

## Test Summary

Requirement 17.3: The system shall return a boolean result (True/False) or
None when the font anomaly status cannot be determined.

Requirement 17.4: The system shall return a confidence score in [0.0, 1.0]
alongside the anomaly result.

Requirement 17.5: The system shall return a human-readable reason string
describing the outcome.

## Properties Tested

- **Property 44a**: FontAnomalyResult always contains required fields with
  correct types (result: Optional[bool], confidence: float, reason: str).

- **Property 44b**: confidence is always in [0.0, 1.0].

- **Property 44c**: When result is None, confidence is 0.0 (cannot determine).

- **Property 44d**: When no image is provided (None), result is None and
  confidence is 0.0.

- **Property 44e**: When image is too small, result is None and confidence
  is 0.0.

- **Property 44f**: For any valid grayscale image, result is True or False
  (never None), and confidence is in [0.0, 1.0].

- **Property 44g**: reason is always a string (never None).
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.grammar_validator import detect_font_anomalies

# ---------------------------------------------------------------------------
# Try importing numpy/cv2 — tests that require image arrays are skipped when
# these are unavailable (mirrors the guard inside detect_font_anomalies).
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def valid_grayscale_image(draw):
    """Generate a valid grayscale numpy array of sufficient size (≥15×40).

    Uses st.binary() to avoid Hypothesis BUFFER_SIZE limits on large lists.
    """
    h = draw(st.integers(min_value=15, max_value=60))
    w = draw(st.integers(min_value=40, max_value=120))
    raw = draw(st.binary(min_size=h * w, max_size=h * w))
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w).copy()


@st.composite
def valid_bgr_image(draw):
    """Generate a valid BGR numpy array of sufficient size (≥15×40).

    Uses st.binary() to avoid Hypothesis BUFFER_SIZE limits on large lists.
    """
    h = draw(st.integers(min_value=15, max_value=60))
    w = draw(st.integers(min_value=40, max_value=120))
    raw = draw(st.binary(min_size=h * w * 3, max_size=h * w * 3))
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()


@st.composite
def too_small_image(draw):
    """Generate a numpy array that is too small for analysis (h<15 or w<40)."""
    too_small_h = draw(st.booleans())
    if too_small_h:
        h = draw(st.integers(min_value=1, max_value=14))
        w = draw(st.integers(min_value=1, max_value=39))
    else:
        h = draw(st.integers(min_value=1, max_value=14))
        w = draw(st.integers(min_value=1, max_value=39))
    raw = draw(st.binary(min_size=h * w, max_size=h * w))
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w).copy()


# ---------------------------------------------------------------------------
# Property 44: Font Anomaly Detection Output
# ---------------------------------------------------------------------------

class TestFontAnomalyDetectionOutputProperties:
    """
    Property-based tests for detect_font_anomalies().

    **Property 44: Font Anomaly Detection Output**
    **Validates: Requirements 17.3, 17.4, 17.5**
    """

    # ------------------------------------------------------------------
    # Property 44a — result always has required fields with correct types
    # ------------------------------------------------------------------

    def test_property_44a_none_input_has_required_fields(self):
        """
        **Property 44a: FontAnomalyResult always contains required fields**

        detect_font_anomalies(None) must return a dict with:
          - result (Optional[bool])
          - confidence (float)
          - reason (str)

        **Validates: Requirements 17.3, 17.4, 17.5**
        """
        result = detect_font_anomalies(None)

        assert "result" in result, "FontAnomalyResult must contain 'result'"
        assert "confidence" in result, "FontAnomalyResult must contain 'confidence'"
        assert "reason" in result, "FontAnomalyResult must contain 'reason'"

        assert result["result"] is None or isinstance(result["result"], bool), (
            f"'result' must be Optional[bool], got {type(result['result'])}"
        )
        assert isinstance(result["confidence"], float), (
            f"'confidence' must be float, got {type(result['confidence'])}"
        )
        assert isinstance(result["reason"], str), (
            f"'reason' must be str, got {type(result['reason'])}"
        )

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    @given(image=valid_grayscale_image())
    @settings(max_examples=100, deadline=10000)
    def test_property_44a_grayscale_image_has_required_fields(self, image):
        """
        **Property 44a: FontAnomalyResult always contains required fields**

        For any valid grayscale image, detect_font_anomalies() must return a
        dict with result (Optional[bool]), confidence (float), reason (str).

        **Validates: Requirements 17.3, 17.4, 17.5**
        """
        result = detect_font_anomalies(image)

        assert "result" in result
        assert "confidence" in result
        assert "reason" in result

        assert result["result"] is None or isinstance(result["result"], bool), (
            f"'result' must be Optional[bool], got {type(result['result'])}"
        )
        assert isinstance(result["confidence"], float), (
            f"'confidence' must be float, got {type(result['confidence'])}"
        )
        assert isinstance(result["reason"], str), (
            f"'reason' must be str, got {type(result['reason'])}"
        )

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    @given(image=valid_bgr_image())
    @settings(max_examples=100, deadline=10000)
    def test_property_44a_bgr_image_has_required_fields(self, image):
        """
        **Property 44a: FontAnomalyResult always contains required fields (BGR)**

        For any valid BGR image, detect_font_anomalies() must return a
        dict with result (Optional[bool]), confidence (float), reason (str).

        **Validates: Requirements 17.3, 17.4, 17.5**
        """
        result = detect_font_anomalies(image)

        assert "result" in result
        assert "confidence" in result
        assert "reason" in result

        assert result["result"] is None or isinstance(result["result"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["reason"], str)

    # ------------------------------------------------------------------
    # Property 44b — confidence is always in [0.0, 1.0]
    # ------------------------------------------------------------------

    def test_property_44b_none_input_confidence_in_range(self):
        """
        **Property 44b: confidence is always in [0.0, 1.0]**

        detect_font_anomalies(None) must return confidence in [0.0, 1.0].

        **Validates: Requirements 17.4**
        """
        result = detect_font_anomalies(None)
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"confidence={result['confidence']} is outside [0.0, 1.0]"
        )

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    @given(image=valid_grayscale_image())
    @settings(max_examples=100, deadline=10000)
    def test_property_44b_grayscale_confidence_in_range(self, image):
        """
        **Property 44b: confidence is always in [0.0, 1.0] for grayscale images**

        **Validates: Requirements 17.4**
        """
        result = detect_font_anomalies(image)
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"confidence={result['confidence']} is outside [0.0, 1.0]"
        )

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    @given(image=valid_bgr_image())
    @settings(max_examples=100, deadline=10000)
    def test_property_44b_bgr_confidence_in_range(self, image):
        """
        **Property 44b: confidence is always in [0.0, 1.0] for BGR images**

        **Validates: Requirements 17.4**
        """
        result = detect_font_anomalies(image)
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"confidence={result['confidence']} is outside [0.0, 1.0]"
        )

    # ------------------------------------------------------------------
    # Property 44c — when result is None, confidence must be 0.0
    # ------------------------------------------------------------------

    def test_property_44c_none_result_has_zero_confidence(self):
        """
        **Property 44c: When result is None, confidence is 0.0**

        When the system cannot determine font anomaly status (result=None),
        it must report confidence=0.0 to avoid misleading callers.

        **Validates: Requirements 17.3, 17.4**
        """
        result = detect_font_anomalies(None)
        assert result["result"] is None
        assert result["confidence"] == 0.0, (
            f"When result is None, confidence must be 0.0, got {result['confidence']}"
        )

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    @given(image=too_small_image())
    @settings(max_examples=100, deadline=5000)
    def test_property_44c_too_small_image_has_zero_confidence(self, image):
        """
        **Property 44c: Too-small images return result=None and confidence=0.0**

        Images smaller than the minimum size (h<15 or w<40) cannot be
        analysed, so result must be None and confidence must be 0.0.

        **Validates: Requirements 17.3, 17.4**
        """
        result = detect_font_anomalies(image)
        assert result["result"] is None, (
            f"Too-small image ({image.shape}) must return result=None, "
            f"got result={result['result']}"
        )
        assert result["confidence"] == 0.0, (
            f"Too-small image must return confidence=0.0, got {result['confidence']}"
        )

    # ------------------------------------------------------------------
    # Property 44d — None input returns result=None, confidence=0.0
    # ------------------------------------------------------------------

    def test_property_44d_none_input_returns_unknown(self):
        """
        **Property 44d: None input returns result=None and confidence=0.0**

        When no image is provided, the system cannot determine font anomaly
        status and must return result=None with confidence=0.0.

        **Validates: Requirements 17.3, 17.4**
        """
        result = detect_font_anomalies(None)
        assert result["result"] is None, (
            f"None input must return result=None, got result={result['result']}"
        )
        assert result["confidence"] == 0.0, (
            f"None input must return confidence=0.0, got {result['confidence']}"
        )
        assert isinstance(result["reason"], str) and len(result["reason"]) > 0, (
            "None input must return a non-empty reason string"
        )

    # ------------------------------------------------------------------
    # Property 44e — too-small images return result=None
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    def test_property_44e_minimum_size_boundary_h14(self):
        """
        **Property 44e: Image with h=14 (below minimum 15) returns result=None**

        **Validates: Requirements 17.3**
        """
        image = np.zeros((14, 40), dtype=np.uint8)
        result = detect_font_anomalies(image)
        assert result["result"] is None, (
            f"14×40 image is below minimum height, must return result=None"
        )

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    def test_property_44e_minimum_size_boundary_w39(self):
        """
        **Property 44e: Image with w=39 (below minimum 40) returns result=None**

        **Validates: Requirements 17.3**
        """
        image = np.zeros((15, 39), dtype=np.uint8)
        result = detect_font_anomalies(image)
        assert result["result"] is None, (
            f"15×39 image is below minimum width, must return result=None"
        )

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    def test_property_44e_minimum_size_boundary_passes(self):
        """
        **Property 44e: Image at exactly minimum size (15×40) does not return None**

        An image of exactly 15×40 meets the minimum size requirement and
        must return result=True or result=False (not None).

        **Validates: Requirements 17.3**
        """
        image = np.zeros((15, 40), dtype=np.uint8)
        result = detect_font_anomalies(image)
        assert result["result"] is not None, (
            f"15×40 image meets minimum size, must return True or False, got None"
        )

    # ------------------------------------------------------------------
    # Property 44f — valid images return True or False (never None)
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    @given(image=valid_grayscale_image())
    @settings(max_examples=100, deadline=10000)
    def test_property_44f_valid_grayscale_returns_bool(self, image):
        """
        **Property 44f: Valid grayscale images return result=True or False**

        For any grayscale image meeting the minimum size requirements,
        detect_font_anomalies() must return result=True or result=False,
        never None.

        **Validates: Requirements 17.3**
        """
        result = detect_font_anomalies(image)
        assert result["result"] is not None, (
            f"Valid {image.shape} grayscale image must return True or False, got None. "
            f"reason='{result['reason']}'"
        )
        assert isinstance(result["result"], bool), (
            f"result must be bool, got {type(result['result'])}"
        )

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    @given(image=valid_bgr_image())
    @settings(max_examples=100, deadline=10000)
    def test_property_44f_valid_bgr_returns_bool(self, image):
        """
        **Property 44f: Valid BGR images return result=True or False**

        For any BGR image meeting the minimum size requirements,
        detect_font_anomalies() must return result=True or result=False,
        never None.

        **Validates: Requirements 17.3**
        """
        result = detect_font_anomalies(image)
        assert result["result"] is not None, (
            f"Valid {image.shape} BGR image must return True or False, got None. "
            f"reason='{result['reason']}'"
        )
        assert isinstance(result["result"], bool), (
            f"result must be bool, got {type(result['result'])}"
        )

    # ------------------------------------------------------------------
    # Property 44g — reason is always a non-None string
    # ------------------------------------------------------------------

    def test_property_44g_none_input_reason_is_string(self):
        """
        **Property 44g: reason is always a string (never None)**

        detect_font_anomalies(None) must return a str reason field.

        **Validates: Requirements 17.5**
        """
        result = detect_font_anomalies(None)
        assert result["reason"] is not None, "reason must not be None"
        assert isinstance(result["reason"], str), (
            f"reason must be str, got {type(result['reason'])}"
        )

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    @given(image=valid_grayscale_image())
    @settings(max_examples=100, deadline=10000)
    def test_property_44g_valid_image_reason_is_string(self, image):
        """
        **Property 44g: reason is always a string for valid images**

        **Validates: Requirements 17.5**
        """
        result = detect_font_anomalies(image)
        assert result["reason"] is not None, "reason must not be None"
        assert isinstance(result["reason"], str), (
            f"reason must be str, got {type(result['reason'])}"
        )

    # ------------------------------------------------------------------
    # Property 44h — result=True implies confidence > 0.0
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    @given(image=valid_grayscale_image())
    @settings(max_examples=100, deadline=10000)
    def test_property_44h_anomaly_detected_implies_positive_confidence(self, image):
        """
        **Property 44h: result=True implies confidence > 0.0**

        When font anomalies are detected (result=True), the confidence score
        must be positive — a zero-confidence anomaly detection is contradictory.

        **Validates: Requirements 17.4**
        """
        result = detect_font_anomalies(image)
        if result["result"] is True:
            assert result["confidence"] > 0.0, (
                f"result=True must have confidence > 0.0, got {result['confidence']}"
            )

    # ------------------------------------------------------------------
    # Spot-checks: known inputs
    # ------------------------------------------------------------------

    def test_property_44_none_input_spot_check(self):
        """
        **Property 44: None input returns expected values**

        detect_font_anomalies(None) must return:
          result=None, confidence=0.0, reason='No image provided'

        **Validates: Requirements 17.3, 17.4, 17.5**
        """
        result = detect_font_anomalies(None)
        assert result["result"] is None
        assert result["confidence"] == 0.0
        assert result["reason"] == "No image provided"

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    def test_property_44_empty_array_returns_unknown(self):
        """
        **Property 44: Empty numpy array returns result=None**

        An array with size=0 cannot be analysed.

        **Validates: Requirements 17.3**
        """
        image = np.array([], dtype=np.uint8).reshape(0, 0)
        result = detect_font_anomalies(image)
        assert result["result"] is None

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    def test_property_44_uniform_white_image_returns_bool(self):
        """
        **Property 44: Uniform white image returns True or False (not None)**

        A 30×80 all-white image is large enough to be analysed.
        The result must be a bool (standard font expected for blank image).

        **Validates: Requirements 17.3**
        """
        image = np.full((30, 80), 255, dtype=np.uint8)
        result = detect_font_anomalies(image)
        assert isinstance(result["result"], bool), (
            f"Uniform white image must return bool, got {result['result']}"
        )
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not available")
    def test_property_44_uniform_black_image_returns_bool(self):
        """
        **Property 44: Uniform black image returns True or False (not None)**

        A 30×80 all-black image is large enough to be analysed.

        **Validates: Requirements 17.3**
        """
        image = np.zeros((30, 80), dtype=np.uint8)
        result = detect_font_anomalies(image)
        assert isinstance(result["result"], bool), (
            f"Uniform black image must return bool, got {result['result']}"
        )
        assert 0.0 <= result["confidence"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
