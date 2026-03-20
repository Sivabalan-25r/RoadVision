"""
Plate Deduplication Logic
"""

from typing import List, Dict, Any
from .levenshtein import levenshtein_distance
import re

def normalize_plate(plate: str) -> str:
    """
    Lowercase and strip spaces/hyphens from a plate string.
    """
    return re.sub(r"[\s\-]+", "", plate.strip().upper())

def deduplicate_detections(detections: List[Dict[str, Any]], distance_threshold: int = 2) -> List[Dict[str, Any]]:
    """
    Deduplicate a list of plate detections using Levenshtein distance and confidence.
    
    1. Group detections by normalized plate string.
    2. Merge groups that have a Levenshtein distance <= distance_threshold 
       AND a length difference <= distance_threshold.
    3. For each merged group, pick the detection with the highest confidence.
    """
    if not detections:
        return []

    # First, separate detections into groups by their normalized plate
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        norm = normalize_plate(det.get('detected_plate', ''))
        if not norm:
            continue
        if norm not in groups:
            groups[norm] = []
        groups[norm].append(det)

    # Sort normalized plates to ensure consistent merging
    sorted_plates = sorted(groups.keys())
    merged_groups: List[set] = []
    
    for plate in sorted_plates:
        found_group = False
        for mg in merged_groups:
            # Check against first plate in the group for simplicity
            base_plate = list(mg)[0]
            if (levenshtein_distance(plate, base_plate) <= distance_threshold and 
                abs(len(plate) - len(base_plate)) <= distance_threshold):
                mg.add(plate)
                found_group = True
                break
        if not found_group:
            merged_groups.append({plate})

    final_detections = []
    for mg in merged_groups:
        all_dets_in_group = []
        for plate in mg:
            all_dets_in_group.extend(groups[plate])
        
        if not all_dets_in_group:
            continue
            
        # Select detection with highest confidence
        best_det = max(all_dets_in_group, key=lambda d: d.get('confidence', 0.0))
        
        # Add frames_seen count
        best_det['frames_seen'] = len(all_dets_in_group)
        
        final_detections.append(best_det)

    return final_detections
