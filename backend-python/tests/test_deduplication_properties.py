"""
Property-Based Tests for Plate Deduplication
"""

import os
import sys
import pytest
from hypothesis import given, strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deduplication.plate_deduplicator import deduplicate_detections

class TestDeduplicationProperties:
    
    @given(detections=st.lists(st.fixed_dictionaries({
        "detected_plate": st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=6, max_size=12),
        "confidence": st.floats(min_value=0.0, max_value=1.0)
    }), min_size=0, max_size=10))
    def test_deduplication_preserves_some_results(self, detections):
        """Deduplication should not return more detections than input."""
        result = deduplicate_detections(detections)
        assert len(result) <= len(detections)
        
    def test_exact_duplicates(self):
        """Exact duplicates should be merged."""
        detections = [
            {"detected_plate": "MH12AB1234", "confidence": 0.8},
            {"detected_plate": "MH12AB1234", "confidence": 0.9},
            {"detected_plate": "MH12AB1234", "confidence": 0.7}
        ]
        result = deduplicate_detections(detections)
        assert len(result) == 1
        assert result[0]["detected_plate"] == "MH12AB1234"
        assert result[0]["confidence"] == 0.9
        assert result[0]["frames_seen"] == 3
        
    def test_fuzzy_duplicates(self):
        """Similar plates should be merged."""
        detections = [
            {"detected_plate": "MH12AB1234", "confidence": 0.8},
            {"detected_plate": "MH12AB1235", "confidence": 0.9}
        ]
        result = deduplicate_detections(detections)
        assert len(result) == 1
        # It picks the one with highest confidence
        assert result[0]["detected_plate"] == "MH12AB1235"
        assert result[0]["confidence"] == 0.9
        assert result[0]["frames_seen"] == 2
        
    def test_distinct_plates(self):
        """Distinct plates should not be merged."""
        detections = [
            {"detected_plate": "MH12AB1234", "confidence": 0.8},
            {"detected_plate": "TN37XY9999", "confidence": 0.9}
        ]
        result = deduplicate_detections(detections)
        assert len(result) == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
