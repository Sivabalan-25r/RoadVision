"""
License Plate Parser Module
"""

import re
from typing import Optional, Dict

# Standard Indian RTO format: SS DD LL NNNN (or SS DD L NNNN)
# Example: MH 12 AB 1234, DL 01 A 0001
STANDARD_PATTERN = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{1,4})$")

# Bharat Series: YY BH NNNN AA
# Example: 22 BH 1234 AA
BHARAT_PATTERN = re.compile(r"^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$")

def parse_plate(plate_text: str) -> Dict[str, Optional[str]]:
    """
    Parse a normalized plate string into its components.
    """
    plate = plate_text.strip().upper().replace(" ", "").replace("-", "")
    
    # Try Bharat Series first
    match = BHARAT_PATTERN.match(plate)
    if match:
        return {
            "type": "BHARAT",
            "year": match.group(1),
            "series": "BH",
            "number": match.group(2),
            "alpha": match.group(3),
            "state": None,
            "district": None
        }
        
    # Try Standard format
    match = STANDARD_PATTERN.match(plate)
    if match:
        return {
            "type": "STANDARD",
            "state": match.group(1),
            "district": match.group(2),
            "series": match.group(3),
            "number": match.group(4),
            "year": None,
            "alpha": None
        }
        
    return {
        "type": "UNKNOWN",
        "raw": plate,
        "state": None,
        "district": None,
        "series": None,
        "number": None,
        "year": None,
        "alpha": None
    }
