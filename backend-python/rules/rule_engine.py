"""
EvasionEye — Cross-Validation Rule Engine
Implements high-level business rules and legal compliance checks
based on vehicle type, zone, and registration status.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PLATE_TYPE_RULES = {
    # Commercial plate rules
    "yellow_in_residential_zone": "ILLEGAL — Commercial vehicle in restricted zone",

    # Temporary plate rules
    "red_plate_repeat_detection": "ILLEGAL — Temporary registration likely expired (seen 3+ times)",

    # Diplomatic handling
    "blue_plate": "DIPLOMATIC — Skip standard DB lookup, log to diplomatic registry",

    # HSRP mandate (mandatory since 6th December 2018)
    "missing_hsrp_permanent": "ILLEGAL — HSRP mandatory for all permanent plates (₹5,000–₹10,000 fine)",

    # Color mismatch
    "wrong_color_combination": "ILLEGAL — Non-standard plate color (RTO Rules 50 & 51 violation)",

    # Rental vehicle
    "black_yellow_commercial_use": "FLAG — Self-drive rental plate, verify rental agreement",

    # EV plates
    "green_plate_commercial": "Commercial EV — verify commercial driving license",
    "green_plate_private": "Private EV — standard compliance check",
}

def apply_business_rules(classification: dict, sightings: int = 0, zone_type: str = "residential") -> List[str]:
    """
    Apply high-level business rules based on plate classification, 
    sighting history, and environmental context (zone).
    """
    violations = []
    p_type = classification.get("plate_type", "Unknown")
    hsrp = classification.get("hsrp_status", "NON-HSRP")
    
    # 1. Commercial in Residential Zone (DISABLED — no zone data available)
    # if p_type == "Commercial/Transport" and zone_type == "residential":
    #     violations.append(PLATE_TYPE_RULES["yellow_in_residential_zone"])

    # 2. Temporary Plate Expiry
    if p_type == "Temporary Registration" and sightings >= 3:
        violations.append(PLATE_TYPE_RULES["red_plate_repeat_detection"])

    # 3. Diplomatic Vehicle
    if p_type == "Diplomatic Vehicle":
        # Note: Handled by caller to skip DB lookup
        violations.append(PLATE_TYPE_RULES["blue_plate"])

    # 4. HSRP Mandate
    if hsrp == "NON-HSRP" and p_type not in ["Temporary Registration", "Unknown"]:
        violations.append(PLATE_TYPE_RULES["missing_hsrp_permanent"])

    # 5. Color Mismatch (Rules 50 & 51)
    p_color = (classification.get("color") or "unknown").lower()
    t_color = (classification.get("text_color") or "unknown").lower()
    
    if p_color == "white" and t_color != "black" and t_color != "unknown":
        violations.append(PLATE_TYPE_RULES["wrong_color_combination"])
    elif p_color == "yellow" and t_color != "black" and t_color != "unknown":
        violations.append(PLATE_TYPE_RULES["wrong_color_combination"])

    # 6. Rental Vehicle
    if p_type == "Self-Drive Rental":
        violations.append(PLATE_TYPE_RULES["black_yellow_commercial_use"])

    # 7. EV Verification
    if p_type == "Electric (Commercial)":
        violations.append(PLATE_TYPE_RULES["green_plate_commercial"])
    elif p_type == "Electric (Private)":
        violations.append(PLATE_TYPE_RULES["green_plate_private"])

    return violations
