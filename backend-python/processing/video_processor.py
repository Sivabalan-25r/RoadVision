"""
RoadVision — Video Processor
Extracts frames from uploaded video, runs YOLO license plate detection
via plate_reader, and assembles detection results.
"""

import logging
import os
import tempfile
from typing import List

import cv2
import numpy as np

from recognition.plate_reader import detect_plates

logger = logging.getLogger(__name__)


def process_video(
    video_path: str,
    frame_interval: int = 5,
) -> List[dict]:
    """
    Process a video file: extract frames, detect plates, return raw detections.

    Args:
        video_path: Path to the video file.
        frame_interval: Analyze every Nth frame (default 5).

    Returns:
        List of dicts with 'bbox', 'crop', 'raw_crop', 'confidence', 'frame_number'
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    all_detections = []
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                # Run YOLO plate detection + geometric filtering + preprocessing
                plate_detections = detect_plates(frame)

                for det in plate_detections:
                    det['frame_number'] = frame_number
                    all_detections.append(det)

            frame_number += 1

    except Exception as e:
        logger.error(f"Error processing video at frame {frame_number}: {e}")
    finally:
        cap.release()

    logger.info(
        f"Processed {frame_number} frames, found {len(all_detections)} plate candidates"
    )
    return all_detections


def save_upload_to_temp(file_bytes: bytes, suffix: str = '.mp4') -> str:
    """Save uploaded file bytes to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()
    return tmp.name
