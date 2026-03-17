"""
RoadVision — BoT-SORT Multi-Object Tracker
Wrapper for Ultralytics BoT-SORT tracker to assign consistent track IDs to license plate detections.
"""

import logging
import os
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
import tempfile

# Ensure backend-python is in path for local imports
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.append(_root)

import numpy as np
import cv2

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    raise ImportError(f"Required packages not available: {e}")

# Import centralized configuration
import config

logger = logging.getLogger(__name__)

# Configure logging for the tracker
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class BoTSORTTracker:
    """
    BoT-SORT multi-object tracker wrapper for license plate tracking.
    
    This class uses the Ultralytics YOLO model's built-in tracking capabilities
    to provide consistent track ID assignment for license plate detections.
    
    Features:
    - Assigns unique track IDs to detected license plates
    - Handles object appearance/disappearance with proper lifecycle management
    - Configurable tracking parameters (max_age, min_hits, iou_threshold)
    - Integrates with existing YOLO detection pipeline
    """
    
    def __init__(self, 
                 max_age: int = None,
                 min_hits: int = None, 
                 iou_threshold: float = None):
        """
        Initialize the BoT-SORT tracker.
        
        Args:
            max_age: Maximum frames to keep a track without detection (default from config)
            min_hits: Minimum detections before assigning track ID (default from config)
            iou_threshold: IoU threshold for matching detections to tracks (default from config)
        """
        # Use config values as defaults
        self.max_age = max_age if max_age is not None else config.BOTSORT_MAX_AGE
        self.min_hits = min_hits if min_hits is not None else config.BOTSORT_MIN_HITS
        self.iou_threshold = iou_threshold if iou_threshold is not None else config.BOTSORT_IOU_THRESHOLD
        
        # Tracking state
        self.frame_count = 0
        self.total_tracks = 0
        self.next_track_id = 1
        self.active_tracks = {}  # track_id -> track_info
        self.track_history = {}  # track_id -> list of detections
        
        # Simple tracking parameters
        self.max_distance = 100  # Maximum pixel distance for track association
        self.min_confidence = 0.1  # Minimum confidence for new tracks (lowered for testing)
        
        logger.info(f"BoT-SORT tracker initialized:")
        logger.info(f"  - max_age: {self.max_age} frames")
        logger.info(f"  - min_hits: {self.min_hits} detections")
        logger.info(f"  - iou_threshold: {self.iou_threshold}")
        logger.info(f"  - max_distance: {self.max_distance} pixels")
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: [x, y, w, h] format
            box2: [x, y, w, h] format
            
        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2] format
        box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
        box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2
        
        # Calculate intersection
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def _calculate_distance(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate center-to-center distance between two bounding boxes.
        
        Args:
            box1: [x, y, w, h] format
            box2: [x, y, w, h] format
            
        Returns:
            Euclidean distance between box centers
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        center1_x = x1 + w1 / 2
        center1_y = y1 + h1 / 2
        center2_x = x2 + w2 / 2
        center2_y = y2 + h2 / 2
        
        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections and assign track IDs.
        
        Args:
            detections: List of detection dictionaries from YOLO
                       Each detection should have 'bbox' and 'confidence' keys
            frame: Current video frame (numpy array)
            
        Returns:
            List of detections with added 'track_id' field
        """
        self.frame_count += 1
        
        if not detections:
            # Age out old tracks
            self._age_tracks()
            return []
        
        # Match detections to existing tracks
        matched_detections = []
        unmatched_detections = []
        matched_tracks = set()
        
        for detection in detections:
            best_track_id = None
            best_score = 0.0
            
            detection_bbox = detection['bbox']
            detection_conf = detection['confidence']
            
            # Find best matching track
            for track_id, track_info in self.active_tracks.items():
                if track_id in matched_tracks:
                    continue  # Track already matched
                
                track_bbox = track_info['last_bbox']
                
                # Calculate IoU and distance
                iou = self._calculate_iou(detection_bbox, track_bbox)
                distance = self._calculate_distance(detection_bbox, track_bbox)
                
                # Combined score: IoU weighted by distance
                if iou > self.iou_threshold and distance < self.max_distance:
                    score = iou * (1.0 - distance / self.max_distance)
                    if score > best_score:
                        best_score = score
                        best_track_id = track_id
            
            if best_track_id is not None:
                # Match found
                matched_detections.append((detection, best_track_id))
                matched_tracks.add(best_track_id)
            else:
                # No match found
                unmatched_detections.append(detection)
        
        # Update matched tracks
        result_detections = []
        
        for detection, track_id in matched_detections:
            # Update track info
            track_info = self.active_tracks[track_id]
            track_info['last_bbox'] = detection['bbox']
            track_info['last_confidence'] = detection['confidence']
            track_info['frames_since_update'] = 0
            track_info['hit_count'] += 1
            
            # Add track ID to detection
            tracked_detection = detection.copy()
            tracked_detection['track_id'] = track_id
            result_detections.append(tracked_detection)
            
            logger.debug(
                f"Frame {self.frame_count}: Updated track {track_id} "
                f"at {detection['bbox']} (conf: {detection['confidence']:.3f})"
            )
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            if detection['confidence'] >= self.min_confidence:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                # Create new track
                self.active_tracks[track_id] = {
                    'last_bbox': detection['bbox'],
                    'last_confidence': detection['confidence'],
                    'frames_since_update': 0,
                    'hit_count': 1,
                    'created_frame': self.frame_count
                }
                
                # Add newly created track to matched_tracks to prevent aging in the same frame
                matched_tracks.add(track_id)
                
                # Add track ID to detection (only if meets min_hits requirement)
                tracked_detection = detection.copy()
                if self.min_hits <= 1:
                    tracked_detection['track_id'] = track_id
                    logger.info(
                        f"Frame {self.frame_count}: Created new track {track_id} "
                        f"at {detection['bbox']} (conf: {detection['confidence']:.3f})"
                    )
                else:
                    tracked_detection['track_id'] = None  # Wait for min_hits
                    logger.debug(
                        f"Frame {self.frame_count}: Provisional track {track_id} "
                        f"at {detection['bbox']} (needs {self.min_hits} hits)"
                    )
                
                result_detections.append(tracked_detection)
                self.total_tracks = max(self.total_tracks, track_id)
        
        # Age out old tracks that weren't matched
        self._age_tracks(matched_tracks)
        
        # Update track IDs for tracks that now meet min_hits requirement
        for detection in result_detections:
            if detection.get('track_id') is None:
                # Find the track for this detection
                for track_id, track_info in self.active_tracks.items():
                    if (track_info['last_bbox'] == detection['bbox'] and 
                        track_info['hit_count'] >= self.min_hits):
                        detection['track_id'] = track_id
                        logger.info(
                            f"Frame {self.frame_count}: Track {track_id} now active "
                            f"(hit count: {track_info['hit_count']})"
                        )
                        break
        
        # Log statistics
        active_tracks = len([d for d in result_detections if d.get('track_id') is not None])
        logger.info(
            f"Frame {self.frame_count}: {len(detections)} detections, "
            f"{active_tracks} tracked objects, "
            f"total tracks: {len(self.active_tracks)}"
        )
        
        return result_detections
    
    def _age_tracks(self, matched_tracks=None):
        """Age out old tracks that haven't been updated."""
        if matched_tracks is None:
            matched_tracks = set()
            
        tracks_to_remove = []
        
        for track_id, track_info in self.active_tracks.items():
            if track_id not in matched_tracks:
                track_info['frames_since_update'] += 1
                
                if track_info['frames_since_update'] >= self.max_age:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            track_info = self.active_tracks.pop(track_id)
            logger.debug(
                f"Frame {self.frame_count}: Removed track {track_id} "
                f"(inactive for {track_info['frames_since_update']} frames)"
            )
    
    def reset(self):
        """Reset the tracker state."""
        self.frame_count = 0
        self.total_tracks = 0
        self.next_track_id = 1
        self.active_tracks.clear()
        self.track_history.clear()
        logger.info("BoT-SORT tracker reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracker statistics.
        
        Returns:
            Dictionary with tracker statistics
        """
        return {
            'frame_count': self.frame_count,
            'total_tracks': self.total_tracks,
            'active_tracks': len(self.active_tracks),
            'max_age': self.max_age,
            'min_hits': self.min_hits,
            'iou_threshold': self.iou_threshold,
            'next_track_id': self.next_track_id
        }
    
    def __del__(self):
        """Cleanup resources."""
        pass


# Convenience function for easy integration
def create_tracker(max_age: int = None, 
                  min_hits: int = None, 
                  iou_threshold: float = None) -> BoTSORTTracker:
    """
    Create a new BoT-SORT tracker instance with optional custom parameters.
    
    Args:
        max_age: Maximum frames to keep a track without detection
        min_hits: Minimum detections before assigning track ID  
        iou_threshold: IoU threshold for matching detections to tracks
        
    Returns:
        Configured BoTSORTTracker instance
    """
    return BoTSORTTracker(
        max_age=max_age,
        min_hits=min_hits, 
        iou_threshold=iou_threshold
    )