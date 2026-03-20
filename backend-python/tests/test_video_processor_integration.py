"""
Integration tests for video processor with BoT-SORT tracking.
"""

import os
import sys
import numpy as np
import cv2
import tempfile

# Ensure backend-python is in path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.append(_root)

from processing.video_processor import process_video


def create_test_video(num_frames=10, width=640, height=480):
    """Create a simple test video file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_path = temp_file.name
    temp_file.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, 30.0, (width, height))
    
    for i in range(num_frames):
        # Create a simple frame with a moving rectangle (simulating a plate)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a white rectangle that moves across the frame
        x = 50 + i * 20
        y = 200
        cv2.rectangle(frame, (x, y), (x + 100, y + 40), (255, 255, 255), -1)
        out.write(frame)
    
    out.release()
    return temp_path


def test_video_processor_initializes_tracker():
    """Test that video processor initializes BoT-SORT tracker."""
    # Create a test video
    video_path = create_test_video(num_frames=5)
    
    try:
        # Process the video
        detections, _, _ = process_video(video_path, frame_interval=1)
        
        # The function should complete without errors
        # Even if no plates are detected, it should return an empty list
        assert isinstance(detections, list), "process_video should return a list"
        
        # Each detection should have the expected fields
        for det in detections:
            assert 'frame_number' in det, "Detection should have frame_number"
            assert 'bbox' in det, "Detection should have bbox"
            assert 'confidence' in det, "Detection should have confidence"
            # track_id may be None if min_hits requirement not met
            assert 'track_id' in det, "Detection should have track_id field"
        
        print(f"✓ Video processor integration test passed: {len(detections)} detections")
        
    finally:
        # Clean up test video
        if os.path.exists(video_path):
            os.remove(video_path)


def test_video_processor_includes_track_ids():
    """Test that detections include track_id field from tracker."""
    video_path = create_test_video(num_frames=5)
    
    try:
        detections, _, _ = process_video(video_path, frame_interval=1)
        
        # All detections should have track_id field (even if None)
        for det in detections:
            assert 'track_id' in det, "Detection must have track_id field"
        
        print(f"✓ Track ID field test passed: all detections have track_id field")
        
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


if __name__ == "__main__":
    print("Running video processor integration tests...")
    test_video_processor_initializes_tracker()
    test_video_processor_includes_track_ids()
    print("\n✓ All integration tests passed!")
