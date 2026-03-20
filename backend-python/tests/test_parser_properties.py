"""
Property-Based Tests for Plate Parser & Pretty-Printer
"""

import os
import sys
import pytest
from hypothesis import given, strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.parser.plate_parser import parse_plate
from rules.parser.pretty_printer import format_plate

class TestParserProperties:
    
    @given(plate=st.from_regex(r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$"))
    def test_standard_parsing(self, plate):
        """Standard plates are parsed correctly."""
        plate = plate.strip()
        parsed = parse_plate(plate)
        assert parsed["type"] == "STANDARD", f"Expected STANDARD type for {plate}, got {parsed['type']}"
        assert parsed["state"] == plate[:2], f"State mismatch for {plate}"
        assert parsed["district"] == plate[2:4], f"District mismatch for {plate}"
        assert parsed["series"] == plate[4:6], f"Series mismatch for {plate}"
        assert parsed["number"] == plate[6:], f"Number mismatch for {plate}"
        
    @given(plate=st.from_regex(r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$"))
    def test_bharat_parsing(self, plate):
        """Bharat series plates are parsed correctly."""
        plate = plate.strip()
        parsed = parse_plate(plate)
        assert parsed["type"] == "BHARAT"
        assert parsed["year"] == plate[:2]
        assert parsed["number"] == plate[4:8]
        assert parsed["alpha"] == plate[8:]

class TestPrettyPrinterProperties:
    
    def test_standard_format(self):
        """Standard plates are formatted with spaces."""
        assert format_plate("MH12AB1234") == "MH 12 AB 1234"
        assert format_plate("DL01A0001") == "DL 01 A 0001"
        
    def test_bharat_format(self):
        """Bharat plates are formatted with spaces."""
        assert format_plate("22BH1234AA") == "22 BH 1234 AA"
        
    @given(plate=st.from_regex(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{1,4}$"))
    def test_idempotence(self, plate):
        """Formatting a normalized plate should be stable (parsing the result gives same info)."""
        plate = plate.strip()
        # We test with strings that match the standard pattern to ensure pretty printing happens
        formatted = format_plate(plate)
        parsed_orig = parse_plate(plate)
        parsed_formatted = parse_plate(formatted)
        
        # We only compare the relevant components that define the identity
        for key in ["type", "state", "district", "series", "number", "year", "series", "alpha"]:
             if key in parsed_orig:
                 assert parsed_orig[key] == parsed_formatted.get(key), f"Mismatch for {key} on plate {plate} (formatted: {formatted})"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
