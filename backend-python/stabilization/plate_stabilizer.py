"""
Plate Stabilization Tracker
"""

import time
from typing import Dict, Any, Optional

class PlateStabilizer:
    """
    Tracks license plates across frames to ensure stability.
    """
    def __init__(self, stabilization_frames: int = 2, expiry_sec: int = 30):
        self.stabilization_frames = stabilization_frames
        self.expiry_sec = expiry_sec
        self.tracker: Dict[str, Dict[str, Any]] = {}

    def stabilize_detection(self, normalized_plate: str, result_entry: Dict[str, Any]) -> bool:
        """
        Track a plate. Returns True if confirmed (seen in >= STABILIZATION_FRAMES frames).
        """
        now = time.time()
        
        # Housekeeping: Expire stale entries
        stale = [k for k, v in self.tracker.items() if now - v['last_seen'] > self.expiry_sec]
        for k in stale:
            del self.tracker[k]
            
        if normalized_plate in self.tracker:
            entry = self.tracker[normalized_plate]
            entry['count'] += 1
            entry['last_seen'] = now
            
            # Keep highest confidence reading
            if result_entry.get('confidence', 0.0) > entry['best_conf']:
                entry['best_conf'] = result_entry.get('confidence', 0.0)
                entry['best_entry'] = result_entry
            
            # Check if confirmed
            if entry['count'] >= self.stabilization_frames:
                # Update result entry with frames_seen
                result_entry['frames_seen'] = entry['count']
                return True
            return False
        else:
            # First sighting
            self.tracker[normalized_plate] = {
                'count': 1,
                'best_conf': result_entry.get('confidence', 0.0),
                'best_entry': result_entry,
                'last_seen': now,
                'saved': False
            }
            # Fast-path: if configured for single-frame stabilization,
            # confirm immediately on first sighting.
            if self.stabilization_frames <= 1:
                result_entry['frames_seen'] = 1
                return True
            return False

    def is_saved(self, normalized_plate: str) -> bool:
        """Check if the plate has already been marked as saved (to DB)."""
        return self.tracker.get(normalized_plate, {}).get('saved', False)

    def mark_saved(self, normalized_plate: str):
        """Mark the plate as saved."""
        if normalized_plate in self.tracker:
            self.tracker[normalized_plate]['saved'] = True
