"""
Property-Based Tests for Levenshtein Distance
"""

import os
import sys
import pytest
from hypothesis import given, strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deduplication.levenshtein import levenshtein_distance

class TestLevenshteinProperties:
    
    @given(s=st.text())
    def test_reflexivity(self, s):
        """Distance between a string and itself is 0."""
        assert levenshtein_distance(s, s) == 0
        
    @given(s1=st.text(), s2=st.text())
    def test_symmetry(self, s1, s2):
        """Distance is symmetric."""
        assert levenshtein_distance(s1, s2) == levenshtein_distance(s2, s1)
        
    @given(s1=st.text(), s2=st.text())
    def test_lower_bound(self, s1, s2):
        """Distance is at least the absolute difference in lengths."""
        assert levenshtein_distance(s1, s2) >= abs(len(s1) - len(s2))
        
    @given(s1=st.text(), s2=st.text())
    def test_upper_bound(self, s1, s2):
        """Distance is at most the maximum of the lengths."""
        assert levenshtein_distance(s1, s2) <= max(len(s1), len(s2))
        
    def test_known_cases(self):
        """Test with some known strings."""
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("flaw", "lawn") == 2
        assert levenshtein_distance("gumbo", "gambol") == 2
        assert levenshtein_distance("MH12AB1234", "MH12AB1235") == 1
        assert levenshtein_distance("MH12AB1234", "MH12AB123") == 1
        assert levenshtein_distance("MH12AB1234", "MH12AC1234") == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
