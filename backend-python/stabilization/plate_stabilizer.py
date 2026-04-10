"""
Plate Stabilization Tracker
"""

import time
from typing import Dict, Any, Optional, List

class PlateStabilizer:
    """
    Tracks license plates across frames to ensure stability.
    """
    def __init__(self, stabilization_frames: int = 2, expiry_sec: int = 30):
        self.stabilization_frames = stabilization_frames
        self.expiry_sec = expiry_sec
        self.tracker: Dict[str, Dict[str, Any]] = {}
        self.track_to_plate: Dict[int, str] = {} # track_id -> normalized_plate
        self.save_history: List[Dict[str, Any]] = [] # List of {plate, ts, bbox}

    def stabilize_detection(self, normalized_plate: str, result_entry: Dict[str, Any]) -> bool:
        """
        Track a plate. Returns True if confirmed (seen in >= STABILIZATION_FRAMES frames).
        """
        now = time.time()
        track_id = result_entry.get('track_id')
        bbox = result_entry.get('bbox')
        
        # Housekeeping
        self.save_history = [h for h in self.save_history if now - h['ts'] < 60]
        stale = [k for k, v in self.tracker.items() if now - v['last_seen'] > self.expiry_sec]
        for k in stale:
            del self.tracker[k]
            for tid, p in list(self.track_to_plate.items()):
                if p == k: del self.track_to_plate[tid]

        # 1. Similarity Check (Text-based)
        if self.is_fuzzy_saved(normalized_plate):
            return False

        # 2. Geometric Overlap Check (Physical-based)
        # If this is the same physical spot as a recent save, it's a duplicate car
        if bbox and self.is_overlap_saved(bbox):
            return False

        if normalized_plate in self.tracker:
            entry = self.tracker[normalized_plate]
            entry['count'] += 1
            entry['last_seen'] = now
            if result_entry.get('confidence', 0.0) > entry['best_conf']:
                entry['best_conf'] = result_entry.get('confidence', 0.0)
                entry['best_entry'] = result_entry
            if track_id is not None:
                self.track_to_plate[track_id] = normalized_plate
            if entry['count'] >= self.stabilization_frames:
                result_entry['frames_seen'] = entry['count']
                return True
            return False
        else:
            self.tracker[normalized_plate] = {
                'count': 1, 'best_conf': result_entry.get('confidence', 0.0),
                'best_entry': result_entry, 'last_seen': now, 'saved': False
            }
            if track_id is not None:
                self.track_to_plate[track_id] = normalized_plate
            if self.stabilization_frames <= 1:
                result_entry['frames_seen'] = 1
                return True
            return False

    def is_overlap_saved(self, bbox: List[float], iou_threshold: float = 0.7) -> bool:
        """Check if bbox overlaps significantly with a recently saved plate."""
        for h in self.save_history:
            h_bbox = h.get('bbox')
            if not h_bbox: continue
            
            # Simple IoU calculation (bbox is [x, y, w, h] as percentages)
            # x1, y1, x2, y2
            ax1, ay1, aw, ah = bbox
            bx1, by1, bw, bh = h_bbox
            ax2, ay2 = ax1 + aw, ay1 + ah
            bx2, by2 = bx1 + bw, by1 + bh

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            area_a = aw * ah
            area_b = bw * bh
            union = area_a + area_b - inter
            
            if union > 0 and (inter / union) > iou_threshold:
                return True
        return False

    def is_fuzzy_saved(self, plate: str, window_sec: int = 10) -> bool:
        from deduplication.levenshtein import levenshtein_distance
        now = time.time()
        for h in self.save_history:
            if now - h['ts'] < window_sec:
                dist = levenshtein_distance(plate, h['plate'])
                if dist <= 1 or (dist / max(len(plate), len(h['plate'])) < 0.2):
                    return True
        return False

    def is_saved(self, normalized_plate: str) -> bool:
        return self.tracker.get(normalized_plate, {}).get('saved', False) or any(h['plate'] == normalized_plate for h in self.save_history)

    def is_track_saved(self, track_id: int) -> bool:
        if track_id not in self.track_to_plate: return False
        plate = self.track_to_plate[track_id]
        return self.is_saved(plate)

    def mark_saved(self, normalized_plate: str, bbox: Optional[List[float]] = None):
        self.save_history.append({'plate': normalized_plate, 'ts': time.time(), 'bbox': bbox})
        if normalized_plate in self.tracker:
            self.tracker[normalized_plate]['saved'] = True
