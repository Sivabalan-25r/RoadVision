from tracking.botsort_tracker import BoTSORTTracker
import numpy as np

tracker = BoTSORTTracker(max_age=1, min_hits=1, iou_threshold=0.3)
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Frame 1: Create track
detection = {
    'bbox': [50, 50, 80, 40],
    'confidence': 0.9,
    'crop': np.zeros((40, 80), dtype=np.uint8),
    'raw_crop': np.zeros((40, 80, 3), dtype=np.uint8)
}

tracked = tracker.update([detection], frame)
print(f'Frame 1: Track ID = {tracked[0]["track_id"]}, Active tracks = {tracker.get_stats()["active_tracks"]}')

# Frame 2: Empty
tracked = tracker.update([], frame)
print(f'Frame 2: Active tracks = {tracker.get_stats()["active_tracks"]}')

# Frame 3: Empty
tracked = tracker.update([], frame)
print(f'Frame 3: Active tracks = {tracker.get_stats()["active_tracks"]}')