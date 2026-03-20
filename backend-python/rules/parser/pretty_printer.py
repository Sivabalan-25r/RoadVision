"""
License Plate Pretty-Printer Module
"""

from .plate_parser import parse_plate

def format_plate(plate_text: str) -> str:
    """
    Format a raw plate string into a human-readable "pretty" format.
    E.g., "MH12AB1234" -> "MH 12 AB 1234"
    """
    parsed = parse_plate(plate_text)
    
    if parsed["type"] == "STANDARD":
        # MH 12 AB 1234
        components = [
            parsed["state"],
            parsed["district"],
            parsed["series"],
            parsed["number"]
        ]
        return " ".join(c for c in components if c)
    
    if parsed["type"] == "BHARAT":
        # 22 BH 1234 AA
        components = [
            parsed["year"],
            "BH",
            parsed["number"],
            parsed["alpha"]
        ]
        return " ".join(c for c in components if c)
        
    # Fallback: Just return uppercase with spaces removed (normalized)
    return plate_text.strip().upper().replace(" ", "").replace("-", "")
