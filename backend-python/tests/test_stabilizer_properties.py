"""
Property-Based Tests for Plate Stabilizer
"""

import os
import sys
import time
import pytest
from hypothesis import given, strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stabilization.plate_stabilizer import PlateStabilizer

class TestPlateStabilizerProperties:
    
    def test_confirmation_requirement(self):
        """Plate requires multiple sightings to be confirmed."""
        stabilizer = PlateStabilizer(stabilization_frames=2)
        plate = "MH12AB1234"
        entry = {"confidence": 0.8}
        
        # First sighting: not confirmed
        assert stabilizer.stabilize_detection(plate, entry) is False
        
        # Second sighting: confirmed
        assert stabilizer.stabilize_detection(plate, entry) is True
        assert entry["frames_seen"] == 2
        
    def test_expiry(self):
        """Stale entries are expired."""
        stabilizer = PlateStabilizer(stabilization_frames=2, expiry_sec=0) # Instant expiry
        plate = "MH12AB1234"
        entry = {"confidence": 0.8}
        
        # First sighting
        stabilizer.stabilize_detection(plate, entry)
        
        # Wait a tiny bit (not strictly necessary with expiry_sec=0 but good for safety)
        time.sleep(0.01)
        
        # Second sighting after expiry should be treated as new (first) sighting
        assert stabilizer.stabilize_detection(plate, entry) is False
        
    def test_best_confidence_retention(self):
        """Tracker maintains the best entry."""
        stabilizer = PlateStabilizer(stabilization_frames=2)
        plate = "MH12AB1234"
        
        stabilizer.stabilize_detection(plate, {"confidence": 0.5})
        stabilizer.stabilize_detection(plate, {"confidence": 0.9})
        
        assert stabilizer.tracker[plate]["best_conf"] == 0.9

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
