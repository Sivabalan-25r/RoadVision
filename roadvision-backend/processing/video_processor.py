"""
RoadVision — Video Processor
Extracts frames from uploaded video, runs YOLO license plate detection,
crops plate regions for OCR, and assembles detection results.
"""

import logging
import os
import tempfile
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---- YOLO Model Loading ----
_yolo_model = None
_USE_CUSTOM_MODEL = False  # Set True when custom license_plate.pt is available


def _get_yolo_model():
    """Load YOLO model for license plate detection."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        # Check for custom license plate model first
        custom_model_path = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'license_plate.pt'
        )

        if os.path.exists(custom_model_path) and _USE_CUSTOM_MODEL:
            logger.info(f"Loading custom YOLO model: {custom_model_path}")
            _yolo_model = YOLO(custom_model_path)
        else:
            # Fallback: use YOLOv8n (general object detection)
            # Class 2 = 'car' in COCO — we'll detect cars and estimate plate regions
            logger.info("Loading YOLOv8n (general detection, will estimate plate regions)")
            _yolo_model = YOLO('yolov8n.pt')

    return _yolo_model


def estimate_plate_bbox(vehicle_bbox: tuple, frame_shape: tuple) -> tuple:
    """
    Estimate the license plate region from a detected vehicle bounding box.
    Plates are typically at the bottom-center of the vehicle.

    Args:
        vehicle_bbox: (x1, y1, x2, y2) of the vehicle
        frame_shape: (height, width) of the frame

    Returns:
        (x, y, w, h) of estimated plate region
    """
    x1, y1, x2, y2 = vehicle_bbox
    vw = x2 - x1
    vh = y2 - y1

    # Plate is roughly in the bottom 20% of the vehicle, center 40%
    plate_x = int(x1 + vw * 0.30)
    plate_y = int(y1 + vh * 0.75)
    plate_w = int(vw * 0.40)
    plate_h = int(vh * 0.15)

    # Clamp to frame bounds
    h_max, w_max = frame_shape[:2]
    plate_x = max(0, min(plate_x, w_max - 1))
    plate_y = max(0, min(plate_y, h_max - 1))
    plate_w = max(10, min(plate_w, w_max - plate_x))
    plate_h = max(5, min(plate_h, h_max - plate_y))

    return plate_x, plate_y, plate_w, plate_h


def detect_plates_in_frame(
    frame: np.ndarray,
    model,
    confidence_threshold: float = 0.4,
) -> List[dict]:
    """
    Run YOLO detection on a single frame and return plate regions.

    Returns:
        List of dicts with keys: 'bbox' (x,y,w,h), 'crop' (numpy), 'confidence'
    """
    results = model(frame, verbose=False, conf=confidence_threshold)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO class IDs for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
            vehicle_classes = {2, 3, 5, 7}

            if cls_id in vehicle_classes and conf >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Estimate plate region from vehicle bbox
                px, py, pw, ph = estimate_plate_bbox(
                    (x1, y1, x2, y2), frame.shape
                )

                # Crop plate region
                crop = frame[py:py + ph, px:px + pw]
                if crop.size == 0:
                    continue

                # Skip crops too small for reliable OCR (min 60×20 px)
                if pw < 60 or ph < 20:
                    continue

                detections.append({
                    'bbox': [px, py, pw, ph],
                    'crop': crop,
                    'confidence': conf,
                    'vehicle_bbox': [x1, y1, x2 - x1, y2 - y1],
                })

    return detections


def process_video(
    video_path: str,
    frame_interval: int = 20,
) -> List[dict]:
    """
    Process a video file: extract frames, detect plates, return raw detections.

    Args:
        video_path: Path to the video file.
        frame_interval: Analyze every Nth frame (default 20).

    Returns:
        List of dicts with 'bbox', 'crop', 'confidence', 'frame_number'
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    model = _get_yolo_model()
    all_detections = []
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                plate_detections = detect_plates_in_frame(frame, model)

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
