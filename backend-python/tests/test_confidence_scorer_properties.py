"""
Property-Based Tests for Confidence Scorer
"""

import os
import sys
import pytest
from hypothesis import given, strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.confidence_scorer import calculate_confidence

class TestConfidenceScorerProperties:
    
    @given(
        yolo_conf=st.floats(min_value=0.0, max_value=1.0),
        ocr_conf=st.floats(min_value=0.0, max_value=1.0),
        is_valid_format=st.booleans(),
        frames_seen=st.integers(min_value=1, max_value=10),
        confidence_modifier=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_confidence_range(self, yolo_conf, ocr_conf, is_valid_format, frames_seen, confidence_modifier):
        """Confidence score is always in [0.0, 1.0]."""
        conf = calculate_confidence(yolo_conf, ocr_conf, is_valid_format, frames_seen, confidence_modifier)
        assert 0.0 <= conf <= 1.0
        
    @given(
        yolo_conf=st.floats(min_value=0.5, max_value=0.9),
        ocr_conf=st.floats(min_value=0.5, max_value=0.9),
        frames_seen=st.integers(min_value=1, max_value=1),
        confidence_modifier=st.floats(min_value=1.0, max_value=1.0)
    )
    def test_format_boost(self, yolo_conf, ocr_conf, frames_seen, confidence_modifier):
        """Valid format gives a higher confidence score."""
        conf_no_boost = calculate_confidence(yolo_conf, ocr_conf, False, frames_seen, confidence_modifier)
        conf_with_boost = calculate_confidence(yolo_conf, ocr_conf, True, frames_seen, confidence_modifier)
        if conf_no_boost < 1.0:
            assert conf_with_boost >= conf_no_boost
            
    @given(
        yolo_conf=st.floats(min_value=0.5, max_value=0.9),
        ocr_conf=st.floats(min_value=0.5, max_value=0.9),
        is_valid_format=st.booleans()
    )
    def test_stability_boost(self, yolo_conf, ocr_conf, is_valid_format):
        """More frames seen gives a higher confidence score."""
        conf1 = calculate_confidence(yolo_conf, ocr_conf, is_valid_format, 1)
        conf2 = calculate_confidence(yolo_conf, ocr_conf, is_valid_format, 2)
        if conf1 < 1.0:
            assert conf2 >= conf1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
