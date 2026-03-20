"""
Confidence Scorer Module
"""

def calculate_confidence(
    yolo_conf: float,
    ocr_conf: float,
    is_valid_format: bool = False,
    frames_seen: int = 1,
    confidence_modifier: float = 1.0
) -> float:
    """
    Calculate combined confidence score for a detection.
    
    Formula: (YOLO_conf × 0.4 + OCR_conf × 0.6) × format_boost × stability_boost
    """
    # Base weighted average
    base_conf = (yolo_conf * 0.4 + ocr_conf * 0.6)
    
    # Format boost (1.15×) for valid Indian format plates
    format_boost = 1.15 if is_valid_format else 1.0
    
    # Stability boost (0.05 per extra frame)
    stability_boost = 1.0 + (max(0, frames_seen - 1) * 0.05)
    
    # Combined score
    combined_score = base_conf * format_boost * stability_boost * confidence_modifier
    
    # Cap at 1.0 and round to 2 decimals
    return float(round(min(1.0, float(combined_score)), 2))
