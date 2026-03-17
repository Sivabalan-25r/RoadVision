"""
Property-Based Tests for OCR Output Format

This module contains property-based tests that verify the PaddleOCR integration
returns the required output format (text and confidence) across all possible
plate image inputs.

**Validates: Requirements 3.2**

## Test Summary

This test suite implements **Property 5: OCR Returns Text and Confidence**
from the design document, which states:

"For any plate image input, the OCR function SHALL return a tuple of
(text: str, confidence: float) where confidence is in the range [0.0, 1.0]."

## Test Coverage

1. **Property-based test with random plate images**: Uses Hypothesis to generate
   random plate-sized images and verifies the output format is always a valid
   (str, float) tuple.

2. **Predefined plate image tests**: Tests with known image types (blank, gray,
   noise, synthetic plate) to verify consistent behavior.

3. **Edge case tests**: Tests with minimal/maximal images to ensure robustness.

## Verified Properties

For each call to `recognize_plate_paddleocr()`, the tests verify:

- **Return type**: Always returns a tuple of exactly 2 elements
- **Text type**: First element is always a str
- **Confidence type**: Second element is always a float
- **Confidence range**: Confidence is always in [0.0, 1.0]
- **Threshold enforcement**: Low-confidence results return ("", 0.0)
- **Empty input handling**: None/empty images return ("", 0.0) gracefully
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
from typing import Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.plate_reader import recognize_plate_paddleocr


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def generate_plate_image(draw):
    """
    Generate a realistic plate-sized image for OCR testing.

    Plate crops are typically small BGR images with aspect ratios between
    1.5 and 7.0 and heights between 20 and 120 pixels.
    """
    width = draw(st.integers(60, 400))
    height = draw(st.integers(20, 120))
    channels = draw(st.sampled_from([1, 3]))  # grayscale or BGR

    if channels == 1:
        shape = (height, width)
    else:
        shape = (height, width, channels)

    image = draw(arrays(dtype=np.uint8, shape=shape, elements=st.integers(0, 255)))
    return image


@st.composite
def generate_synthetic_plate_image(draw):
    """
    Generate a synthetic plate image with white background and dark characters.

    These images are more likely to produce actual OCR results, allowing
    the property to be tested against non-empty outputs as well.
    """
    width = draw(st.integers(120, 320))
    height = draw(st.integers(40, 80))

    # White background
    image = np.full((height, width, 3), 240, dtype=np.uint8)

    # Add dark rectangular character-like blocks
    num_chars = draw(st.integers(4, 10))
    char_width = max(8, width // (num_chars + 2))
    char_height = max(10, height - 10)

    for i in range(num_chars):
        x = 5 + i * (char_width + 3)
        y = 5
        if x + char_width < width and y + char_height < height:
            darkness = draw(st.integers(0, 80))
            image[y : y + char_height, x : x + char_width] = darkness

    return image


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestOCROutputFormatProperties:
    """
    Property-based tests for OCR output format.

    Verifies that recognize_plate_paddleocr() always returns a valid
    (str, float) tuple regardless of the input image.
    """

    @given(image=generate_plate_image())
    @settings(max_examples=30, deadline=60000)
    def test_property_5_ocr_returns_text_and_confidence(self, image: np.ndarray):
        """
        **Property 5: OCR Returns Text and Confidence**

        For any plate image, recognize_plate_paddleocr() SHALL return a tuple
        of (text: str, confidence: float) where confidence is in [0.0, 1.0].

        **Validates: Requirements 3.2**
        """
        result = recognize_plate_paddleocr(image)

        # Property: must return a tuple of exactly 2 elements
        assert isinstance(result, tuple), "OCR must return a tuple"
        assert len(result) == 2, "OCR tuple must have exactly 2 elements (text, confidence)"

        text, confidence = result

        # Property: text must be a str
        assert isinstance(text, str), f"OCR text must be str, got {type(text)}"

        # Property: confidence must be a float
        assert isinstance(confidence, (float, np.floating)), (
            f"OCR confidence must be float, got {type(confidence)}"
        )

        # Property: confidence must be in [0.0, 1.0]
        assert 0.0 <= confidence <= 1.0, (
            f"OCR confidence {confidence} must be in range [0.0, 1.0]"
        )

        # Property: if confidence is 0.0, text must be empty (threshold enforcement)
        if confidence == 0.0:
            assert text == "", (
                f"When confidence is 0.0, text must be empty string, got '{text}'"
            )

        # Property: if text is non-empty, confidence must be > 0.0
        if text != "":
            assert confidence > 0.0, (
                f"Non-empty text '{text}' must have confidence > 0.0, got {confidence}"
            )

    @given(image=generate_synthetic_plate_image())
    @settings(max_examples=20, deadline=60000)
    def test_property_5_with_synthetic_plates(self, image: np.ndarray):
        """
        Property 5 with synthetic plate images that are more likely to produce
        actual OCR results, exercising the non-empty output path.

        **Validates: Requirements 3.2**
        """
        result = recognize_plate_paddleocr(image)

        assert isinstance(result, tuple) and len(result) == 2
        text, confidence = result

        assert isinstance(text, str)
        assert isinstance(confidence, (float, np.floating))
        assert 0.0 <= confidence <= 1.0

        # Consistency: text/confidence must agree
        assert (text == "") == (confidence == 0.0), (
            f"text and confidence must be consistent: text='{text}', confidence={confidence}"
        )

    def test_property_5_with_predefined_images(self):
        """
        Property 5 with a set of predefined plate images covering common cases.

        **Validates: Requirements 3.2**
        """
        test_images = [
            # Black plate
            np.zeros((40, 120, 3), dtype=np.uint8),
            # White plate
            np.full((40, 120, 3), 255, dtype=np.uint8),
            # Gray plate
            np.full((40, 120, 3), 128, dtype=np.uint8),
            # Random noise
            np.random.randint(0, 256, (40, 120, 3), dtype=np.uint8),
            # Grayscale image
            np.zeros((40, 120), dtype=np.uint8),
            # Minimal size
            np.zeros((20, 60, 3), dtype=np.uint8),
            # Larger plate crop
            np.random.randint(0, 256, (80, 320, 3), dtype=np.uint8),
        ]

        for i, image in enumerate(test_images):
            result = recognize_plate_paddleocr(image)

            assert isinstance(result, tuple), f"Image {i}: must return tuple"
            assert len(result) == 2, f"Image {i}: tuple must have 2 elements"

            text, confidence = result

            assert isinstance(text, str), f"Image {i}: text must be str"
            assert isinstance(confidence, (float, np.floating)), (
                f"Image {i}: confidence must be float"
            )
            assert 0.0 <= confidence <= 1.0, (
                f"Image {i}: confidence {confidence} out of range [0.0, 1.0]"
            )

    def test_property_5_empty_image_returns_empty_tuple(self):
        """
        Property 5 edge case: empty/None image must return ("", 0.0) gracefully.

        **Validates: Requirements 3.2**
        """
        # Empty numpy array
        empty_image = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        result = recognize_plate_paddleocr(empty_image)

        assert result == ("", 0.0), (
            f"Empty image must return ('', 0.0), got {result}"
        )

    def test_property_5_none_image_returns_empty_tuple(self):
        """
        Property 5 edge case: None input must return ("", 0.0) gracefully.

        **Validates: Requirements 3.2**
        """
        result = recognize_plate_paddleocr(None)  # type: ignore[arg-type]

        assert result == ("", 0.0), (
            f"None image must return ('', 0.0), got {result}"
        )

    def test_property_5_confidence_threshold_enforced(self):
        """
        Property 5: results below OCR_CONFIDENCE_THRESHOLD must return ("", 0.0).

        Verifies that the threshold gate in recognize_plate_paddleocr() is
        working — any result with confidence < 0.25 must be suppressed.

        **Validates: Requirements 3.2, 3.3**
        """
        import config

        # A completely black image is very unlikely to produce a high-confidence
        # OCR result; if it does produce one it must still satisfy the format contract.
        black_plate = np.zeros((40, 120, 3), dtype=np.uint8)
        result = recognize_plate_paddleocr(black_plate)

        text, confidence = result

        # Whatever the result, the format contract must hold
        assert isinstance(text, str)
        assert isinstance(confidence, (float, np.floating))
        assert 0.0 <= confidence <= 1.0

        # If confidence is non-zero it must be at or above the threshold
        if confidence > 0.0:
            assert confidence >= config.OCR_CONFIDENCE_THRESHOLD, (
                f"Returned confidence {confidence} is below threshold "
                f"{config.OCR_CONFIDENCE_THRESHOLD}"
            )


class TestOCRConfidenceThresholdingProperties:
    """
    Property-based tests for OCR confidence thresholding.

    Verifies that recognize_plate_paddleocr() correctly applies the
    0.25 minimum confidence threshold: results below are rejected,
    results at or above are accepted.

    **Validates: Requirements 3.3, 3.4**
    """

    @given(
        text=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Nd")),
            min_size=1,
            max_size=12,
        ),
        confidence=st.floats(min_value=0.0, max_value=0.2499, allow_nan=False),
    )
    @settings(max_examples=50, deadline=10000)
    def test_property_6_below_threshold_rejected(self, text: str, confidence: float):
        """
        **Property 6: OCR Confidence Thresholding — below threshold rejected**

        For any OCR result whose raw confidence is strictly below 0.25,
        recognize_plate_paddleocr() SHALL return ("", 0.0).

        **Validates: Requirements 3.3, 3.4**
        """
        import recognition.plate_reader as plate_reader_module
        from unittest.mock import MagicMock, patch

        # Build a fake PaddleOCR result with the given text and confidence
        fake_result = [[[None, (text, confidence)]]]

        mock_ocr = MagicMock()
        mock_ocr.ocr.return_value = fake_result

        with patch.object(plate_reader_module, "_paddleocr_instance", mock_ocr):
            image = np.zeros((40, 120, 3), dtype=np.uint8)
            result = plate_reader_module.recognize_plate_paddleocr(image)

        assert result == ("", 0.0), (
            f"Confidence {confidence:.4f} is below threshold 0.25 — "
            f"expected ('', 0.0) but got {result}"
        )

    @given(
        text=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Nd")),
            min_size=1,
            max_size=12,
        ),
        confidence=st.floats(min_value=0.25, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=10000)
    def test_property_6_at_or_above_threshold_accepted(self, text: str, confidence: float):
        """
        **Property 6: OCR Confidence Thresholding — at/above threshold accepted**

        For any OCR result whose raw confidence is >= 0.25,
        recognize_plate_paddleocr() SHALL return (text, confidence) unchanged.

        **Validates: Requirements 3.3, 3.4**
        """
        import recognition.plate_reader as plate_reader_module
        from unittest.mock import MagicMock, patch

        fake_result = [[[None, (text, confidence)]]]

        mock_ocr = MagicMock()
        mock_ocr.ocr.return_value = fake_result

        with patch.object(plate_reader_module, "_paddleocr_instance", mock_ocr):
            image = np.zeros((40, 120, 3), dtype=np.uint8)
            result = plate_reader_module.recognize_plate_paddleocr(image)

        returned_text, returned_conf = result

        assert returned_text == text, (
            f"Confidence {confidence:.4f} >= 0.25 — expected text '{text}' "
            f"but got '{returned_text}'"
        )
        assert returned_conf == confidence, (
            f"Confidence {confidence:.4f} >= 0.25 — expected confidence "
            f"{confidence} but got {returned_conf}"
        )

    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=10000)
    def test_property_6_threshold_boundary(self, confidence: float):
        """
        **Property 6: OCR Confidence Thresholding — boundary invariant**

        The threshold boundary at 0.25 must be respected consistently:
        - confidence < 0.25  → result is ("", 0.0)
        - confidence >= 0.25 → result is (text, confidence) with confidence >= 0.25

        **Validates: Requirements 3.3, 3.4**
        """
        import config
        import recognition.plate_reader as plate_reader_module
        from unittest.mock import MagicMock, patch

        sample_text = "AB12CD3456"
        fake_result = [[[None, (sample_text, confidence)]]]

        mock_ocr = MagicMock()
        mock_ocr.ocr.return_value = fake_result

        with patch.object(plate_reader_module, "_paddleocr_instance", mock_ocr):
            image = np.zeros((40, 120, 3), dtype=np.uint8)
            result = plate_reader_module.recognize_plate_paddleocr(image)

        returned_text, returned_conf = result

        if confidence < config.OCR_CONFIDENCE_THRESHOLD:
            assert result == ("", 0.0), (
                f"confidence={confidence:.4f} < threshold={config.OCR_CONFIDENCE_THRESHOLD} "
                f"must be rejected, got {result}"
            )
        else:
            assert returned_conf >= config.OCR_CONFIDENCE_THRESHOLD, (
                f"Accepted result confidence {returned_conf:.4f} must be "
                f">= threshold {config.OCR_CONFIDENCE_THRESHOLD}"
            )
            assert returned_text == sample_text, (
                f"Accepted result text must be '{sample_text}', got '{returned_text}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Property 7: Ensemble OCR Selects Highest Confidence
# ---------------------------------------------------------------------------

class TestEnsembleOCRSelectionProperties:
    """
    Property-based tests for ensemble OCR variant selection.

    Verifies that the ensemble OCR pipeline (preprocess_plate_variants +
    _run_ocr_with_fallback) always selects the variant with the highest
    confidence score.

    **Validates: Requirements 3.5**
    """

    @given(
        texts=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Nd")),
                min_size=1,
                max_size=12,
            ),
            min_size=1,
            max_size=4,
        ),
        confidences=st.lists(
            st.floats(min_value=0.25, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=4,
        ),
    )
    @settings(max_examples=50, deadline=10000)
    def test_property_7_ensemble_selects_highest_confidence(
        self, texts: list, confidences: list
    ):
        """
        **Property 7: Ensemble OCR Selects Highest Confidence**

        When multiple preprocessing variants produce OCR results, the ensemble
        SHALL select the variant with the highest confidence score.

        **Validates: Requirements 3.5**
        """
        import recognition.plate_reader as plate_reader_module
        from unittest.mock import patch, MagicMock

        # Align lists to same length
        n = min(len(texts), len(confidences))
        texts = texts[:n]
        confidences = confidences[:n]

        if n == 0:
            return

        # Build fake OCR results for each variant
        call_results = [(t, c) for t, c in zip(texts, confidences)]
        call_iter = iter(call_results)

        def fake_ocr(img):
            try:
                return next(call_iter)
            except StopIteration:
                return ("", 0.0)

        # Patch preprocess_plate_variants to return n dummy images
        dummy_variants = [np.zeros((40, 120, 3), dtype=np.uint8) for _ in range(n)]

        with patch.object(
            plate_reader_module, "preprocess_plate_variants", return_value=dummy_variants
        ), patch.object(
            plate_reader_module, "_run_ocr_with_fallback", side_effect=fake_ocr
        ), patch.object(
            plate_reader_module, "is_garbage_text", return_value=False
        ), patch.object(
            plate_reader_module, "clean_text", side_effect=lambda t: t
        ), patch.object(
            plate_reader_module, "_apply_position_based_corrections", side_effect=lambda t: t
        ):
            image = np.zeros((40, 120, 3), dtype=np.uint8)
            result = plate_reader_module.read_plate(image)

        # The result must be the text with the highest confidence
        expected_text = texts[confidences.index(max(confidences))]
        assert result == expected_text, (
            f"Ensemble must select highest confidence variant. "
            f"Expected '{expected_text}' (conf={max(confidences):.3f}), "
            f"got '{result}'"
        )

    def test_property_7_ensemble_returns_none_when_all_variants_fail(self):
        """
        **Property 7: Ensemble OCR — all variants fail returns None**

        When all preprocessing variants produce no usable OCR result,
        read_plate() SHALL return None.

        **Validates: Requirements 3.5**
        """
        import recognition.plate_reader as plate_reader_module
        from unittest.mock import patch

        dummy_variants = [np.zeros((40, 120, 3), dtype=np.uint8) for _ in range(4)]

        with patch.object(
            plate_reader_module, "preprocess_plate_variants", return_value=dummy_variants
        ), patch.object(
            plate_reader_module, "_run_ocr_with_fallback", return_value=("", 0.0)
        ):
            image = np.zeros((40, 120, 3), dtype=np.uint8)
            result = plate_reader_module.read_plate(image)

        assert result is None, (
            f"When all variants fail, read_plate() must return None, got '{result}'"
        )

    def test_property_7_ensemble_generates_four_variants(self):
        """
        **Property 7: Ensemble OCR generates exactly 4 preprocessing variants**

        preprocess_plate_variants() SHALL return exactly 4 variants:
        original, CLAHE, sharpened, thresholded.

        **Validates: Requirements 3.5, 7.7**
        """
        from recognition.plate_reader import preprocess_plate_variants

        plate = np.random.randint(0, 256, (60, 200, 3), dtype=np.uint8)
        variants = preprocess_plate_variants(plate)

        assert len(variants) == 4, (
            f"preprocess_plate_variants() must return exactly 4 variants, "
            f"got {len(variants)}"
        )

        for i, variant in enumerate(variants):
            assert isinstance(variant, np.ndarray), (
                f"Variant {i} must be a numpy array"
            )
            assert variant.ndim in (2, 3), (
                f"Variant {i} must be 2D (grayscale) or 3D (BGR), got ndim={variant.ndim}"
            )
            assert variant.size > 0, f"Variant {i} must not be empty"

    @given(image=generate_plate_image())
    @settings(max_examples=20, deadline=30000)
    def test_property_7_variants_always_four(self, image: np.ndarray):
        """
        **Property 7: preprocess_plate_variants always returns 4 variants**

        For any plate image input, preprocess_plate_variants() SHALL always
        return a list of exactly 4 numpy arrays.

        **Validates: Requirements 3.5, 7.7**
        """
        from recognition.plate_reader import preprocess_plate_variants

        variants = preprocess_plate_variants(image)

        assert isinstance(variants, list), "preprocess_plate_variants must return a list"
        assert len(variants) == 4, (
            f"preprocess_plate_variants must return exactly 4 variants, got {len(variants)}"
        )
        for i, v in enumerate(variants):
            assert isinstance(v, np.ndarray), f"Variant {i} must be np.ndarray"
            assert v.size > 0, f"Variant {i} must not be empty"


# ---------------------------------------------------------------------------
# Property 18: Preprocessing Output Dimensions
# ---------------------------------------------------------------------------

class TestPreprocessingOutputDimensionsProperties:
    """
    Property-based tests for preprocessing output dimensions.

    Verifies that preprocess_plate_crop() always returns an image of
    exactly 320×120 pixels regardless of the input size or content.

    **Property 18: Preprocessing Output Dimensions**
    **Validates: Requirements 7.6**
    """

    @given(image=generate_plate_image())
    @settings(max_examples=50, deadline=30000)
    def test_property_18_preprocess_plate_crop_output_dimensions(
        self, image: np.ndarray
    ):
        """
        **Property 18: Preprocessing Output Dimensions**

        For any plate image input, preprocess_plate_crop() SHALL return
        an image with exactly width=320 and height=120 pixels.

        **Validates: Requirements 7.6**
        """
        from recognition.plate_reader import preprocess_plate_crop

        result = preprocess_plate_crop(image)

        assert isinstance(result, np.ndarray), (
            "preprocess_plate_crop must return a numpy array"
        )
        assert result.ndim in (2, 3), (
            f"Output must be 2D (grayscale) or 3D (BGR), got ndim={result.ndim}"
        )

        h, w = result.shape[:2]
        assert w == 320, (
            f"Output width must be 320, got {w} (input shape: {image.shape})"
        )
        assert h == 120, (
            f"Output height must be 120, got {h} (input shape: {image.shape})"
        )

    def test_property_18_with_predefined_sizes(self):
        """
        Property 18 with a range of predefined input sizes covering
        small, typical, and large plate crops.

        **Validates: Requirements 7.6**
        """
        from recognition.plate_reader import preprocess_plate_crop

        input_shapes = [
            # Tiny crops
            (15, 50, 3),
            (20, 60, 3),
            # Typical plate crops
            (40, 120, 3),
            (60, 200, 3),
            (80, 320, 3),
            # Large crops
            (120, 400, 3),
            (200, 640, 3),
            # Grayscale inputs
            (40, 120),
            (60, 200),
            # Already at target size
            (120, 320, 3),
            (120, 320),
        ]

        for shape in input_shapes:
            image = np.random.randint(0, 256, shape, dtype=np.uint8)
            result = preprocess_plate_crop(image)

            h, w = result.shape[:2]
            assert w == 320, (
                f"Input shape {shape}: output width must be 320, got {w}"
            )
            assert h == 120, (
                f"Input shape {shape}: output height must be 120, got {h}"
            )

    @given(
        width=st.integers(min_value=30, max_value=800),
        height=st.integers(min_value=10, max_value=300),
        channels=st.sampled_from([1, 3]),
    )
    @settings(max_examples=50, deadline=30000)
    def test_property_18_output_dimensions_independent_of_input_size(
        self, width: int, height: int, channels: int
    ):
        """
        **Property 18: Output dimensions are always 320×120**

        The output dimensions of preprocess_plate_crop() must be exactly
        320×120 regardless of the input width, height, or channel count.

        **Validates: Requirements 7.6**
        """
        from recognition.plate_reader import preprocess_plate_crop

        if channels == 1:
            image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        else:
            image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

        result = preprocess_plate_crop(image)

        out_h, out_w = result.shape[:2]
        assert out_w == 320, (
            f"Output width must always be 320, got {out_w} "
            f"(input: {width}×{height}, channels={channels})"
        )
        assert out_h == 120, (
            f"Output height must always be 120, got {out_h} "
            f"(input: {width}×{height}, channels={channels})"
        )

    def test_property_18_output_is_grayscale(self):
        """
        Property 18 corollary: preprocess_plate_crop() output is always
        a 2D grayscale image (single channel), since the pipeline converts
        to grayscale and applies thresholding.

        **Validates: Requirements 7.6**
        """
        from recognition.plate_reader import preprocess_plate_crop

        # BGR input
        bgr_image = np.random.randint(0, 256, (60, 200, 3), dtype=np.uint8)
        result_bgr = preprocess_plate_crop(bgr_image)
        assert result_bgr.ndim == 2, (
            f"preprocess_plate_crop output must be 2D grayscale, got ndim={result_bgr.ndim}"
        )

        # Grayscale input
        gray_image = np.random.randint(0, 256, (60, 200), dtype=np.uint8)
        result_gray = preprocess_plate_crop(gray_image)
        assert result_gray.ndim == 2, (
            f"preprocess_plate_crop output must be 2D grayscale, got ndim={result_gray.ndim}"
        )

    def test_property_18_output_dtype_is_uint8(self):
        """
        Property 18 corollary: preprocess_plate_crop() output dtype is
        always uint8 (pixel values in [0, 255]).

        **Validates: Requirements 7.6**
        """
        from recognition.plate_reader import preprocess_plate_crop

        image = np.random.randint(0, 256, (60, 200, 3), dtype=np.uint8)
        result = preprocess_plate_crop(image)

        assert result.dtype == np.uint8, (
            f"Output dtype must be uint8, got {result.dtype}"
        )
        assert result.min() >= 0 and result.max() <= 255, (
            f"Output pixel values must be in [0, 255], "
            f"got min={result.min()}, max={result.max()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Property 19: Preprocessing Generates Four Variants
# ---------------------------------------------------------------------------

class TestPreprocessingVariantGenerationProperties:
    """
    Property-based tests for preprocessing variant generation.

    Verifies that preprocess_plate_variants() always generates exactly 4
    distinct preprocessing variants for any plate image input.

    **Property 19: Preprocessing Generates Four Variants**
    **Validates: Requirements 7.7**
    """

    @given(image=generate_plate_image())
    @settings(max_examples=50, deadline=30000)
    def test_property_19_always_generates_four_variants(self, image: np.ndarray):
        """
        **Property 19: Preprocessing Generates Four Variants**

        For any plate image input, preprocess_plate_variants() SHALL return
        a list of exactly 4 numpy arrays representing:
          [0] Original (contrast-normalized grayscale)
          [1] CLAHE-enhanced
          [2] Sharpened + CLAHE
          [3] Adaptive threshold (binary)

        **Validates: Requirements 7.7**
        """
        from recognition.plate_reader import preprocess_plate_variants

        variants = preprocess_plate_variants(image)

        assert isinstance(variants, list), (
            "preprocess_plate_variants must return a list"
        )
        assert len(variants) == 4, (
            f"preprocess_plate_variants must return exactly 4 variants, "
            f"got {len(variants)} (input shape: {image.shape})"
        )

    @given(image=generate_plate_image())
    @settings(max_examples=50, deadline=30000)
    def test_property_19_all_variants_are_valid_arrays(self, image: np.ndarray):
        """
        **Property 19: All variants are valid non-empty numpy arrays**

        Each of the 4 variants SHALL be a non-empty numpy array with
        dtype uint8 and either 2D (grayscale) or 3D (BGR) shape.

        **Validates: Requirements 7.7**
        """
        from recognition.plate_reader import preprocess_plate_variants

        variants = preprocess_plate_variants(image)

        for i, variant in enumerate(variants):
            assert isinstance(variant, np.ndarray), (
                f"Variant {i} must be a numpy array, got {type(variant)}"
            )
            assert variant.size > 0, (
                f"Variant {i} must not be empty (input shape: {image.shape})"
            )
            assert variant.ndim in (2, 3), (
                f"Variant {i} must be 2D or 3D, got ndim={variant.ndim}"
            )
            assert variant.dtype == np.uint8, (
                f"Variant {i} must have dtype uint8, got {variant.dtype}"
            )

    @given(image=generate_plate_image())
    @settings(max_examples=50, deadline=30000)
    def test_property_19_variants_share_spatial_dimensions(self, image: np.ndarray):
        """
        **Property 19: All variants share the same spatial dimensions**

        All 4 variants produced from the same input SHALL have identical
        height and width, since they are derived from the same base image.

        **Validates: Requirements 7.7**
        """
        from recognition.plate_reader import preprocess_plate_variants

        variants = preprocess_plate_variants(image)

        assert len(variants) == 4
        h0, w0 = variants[0].shape[:2]

        for i, variant in enumerate(variants[1:], start=1):
            h, w = variant.shape[:2]
            assert h == h0 and w == w0, (
                f"Variant {i} dimensions ({w}×{h}) differ from variant 0 "
                f"({w0}×{h0}) — all variants must share spatial dimensions"
            )

    @given(image=generate_plate_image())
    @settings(max_examples=30, deadline=30000)
    def test_property_19_variant_4_is_binary(self, image: np.ndarray):
        """
        **Property 19: Variant 4 (index 3) is a binary thresholded image**

        The fourth variant SHALL be the result of adaptive thresholding,
        meaning its pixel values are exclusively 0 or 255.

        **Validates: Requirements 7.7**
        """
        from recognition.plate_reader import preprocess_plate_variants

        variants = preprocess_plate_variants(image)

        thresh_variant = variants[3]
        unique_values = np.unique(thresh_variant)

        assert set(unique_values).issubset({0, 255}), (
            f"Variant 3 (adaptive threshold) must only contain 0 and 255, "
            f"got unique values: {unique_values.tolist()}"
        )

    def test_property_19_with_predefined_plate_sizes(self):
        """
        **Property 19: Four variants generated for all common plate sizes**

        preprocess_plate_variants() SHALL return 4 variants for a range
        of typical plate crop sizes encountered in real-world video frames.

        **Validates: Requirements 7.7**
        """
        from recognition.plate_reader import preprocess_plate_variants

        shapes = [
            (30, 90, 3),    # Small plate
            (40, 120, 3),   # Typical small
            (60, 200, 3),   # Typical medium
            (80, 320, 3),   # Typical large
            (120, 400, 3),  # High-res plate
            (40, 120),      # Grayscale input
            (60, 200),      # Grayscale medium
        ]

        for shape in shapes:
            image = np.random.randint(0, 256, shape, dtype=np.uint8)
            variants = preprocess_plate_variants(image)

            assert len(variants) == 4, (
                f"Input shape {shape}: expected 4 variants, got {len(variants)}"
            )
            for i, v in enumerate(variants):
                assert isinstance(v, np.ndarray) and v.size > 0, (
                    f"Input shape {shape}, variant {i}: must be non-empty ndarray"
                )

    @given(
        width=st.integers(min_value=50, max_value=500),
        height=st.integers(min_value=15, max_value=150),
    )
    @settings(max_examples=30, deadline=30000)
    def test_property_19_count_independent_of_input_dimensions(
        self, width: int, height: int
    ):
        """
        **Property 19: Variant count is always 4 regardless of input dimensions**

        The number of variants returned by preprocess_plate_variants() SHALL
        always be exactly 4, independent of the input image width or height.

        **Validates: Requirements 7.7**
        """
        from recognition.plate_reader import preprocess_plate_variants

        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        variants = preprocess_plate_variants(image)

        assert len(variants) == 4, (
            f"Input {width}×{height}: expected 4 variants, got {len(variants)}"
        )
