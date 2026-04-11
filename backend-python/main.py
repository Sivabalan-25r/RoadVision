import sys
import asyncio

# Fix for Python 3.14 asyncio CancelledError on Windows with Uvicorn
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

"""
EvasionEye — FastAPI Backend
Main application entry point.

Provides the /analyze-video endpoint that processes uploaded traffic videos,
detects number plates using a dedicated YOLOv26 plate detector + PaddleOCR,
validates plate formats against Indian RTO rules, and returns structured
detection results.

Run with: uvicorn main:app --reload
API docs: http://localhost:8000/docs
"""

import logging
import time
import re
import os
import sys
import json
import tempfile
from typing import List, Optional, Union

# Ensure current directory is in path for local imports
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.append(_root)

try:
    import cv2      # type: ignore
    import numpy as np  # type: ignore
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Path, Request # type: ignore
    from fastapi.middleware.cors import CORSMiddleware # type: ignore
    from fastapi.responses import JSONResponse # type: ignore
    from fastapi.staticfiles import StaticFiles # type: ignore
    from pydantic import BaseModel # type: ignore
    from contextlib import asynccontextmanager
except ImportError:
    pass

from processing.video_processor import process_video, save_upload_to_temp
from recognition.plate_reader import (
    read_plate, get_last_ocr_violation, update_tracker_ocr, 
    vote_stabilized_plate, load_plate_model, detect_plates
)
from rules.plate_rules import validate_plate, PLATE_PATTERN
import database as db
from scoring.confidence_scorer import calculate_confidence
from deduplication.levenshtein import levenshtein_distance
from deduplication.plate_deduplicator import deduplicate_detections
from rules.parser.pretty_printer import format_plate
from stabilization.plate_stabilizer import PlateStabilizer
from collections import deque


# ---- Helper Functions ----
# ---- Global Instances ----
plate_stabilizer = PlateStabilizer(stabilization_frames=2, expiry_sec=30)

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evasioneye")

# ---- Lifespan Manager ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events gracefully.
    Replaces deprecated @app.on_event patterns.
    """
    # ---- Startup Logic ----
    # Initialize database
    db.init_database()
    logger.info("Database initialized")
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    detector_path = os.path.join(models_dir, 'yolov26-license-plate.pt')

    logger.info("=" * 60)
    logger.info("EvasionEye Backend v2.0 — Startup")
    logger.info("=" * 60)

    # ---- Load YOLO plate detector (REQUIRED) ----
    try:
        load_plate_model()
    except FileNotFoundError as e:
        logger.error(f"  ✗ FATAL: {e}")
        logger.error("=" * 60)
        raise RuntimeError(f"Plate detector model missing at: {detector_path}")

    # ---- Warm up OCR engines ----
    try:
        from recognition.plate_reader import load_easyocr_model, load_paddleocr_model
        load_paddleocr_model()
        load_easyocr_model()
    except Exception as e:
        logger.warning(f"OCR warmup warning: {e}")

    logger.info("  ✓ System integrity check completed")
    logger.info("=" * 60)
    
    try:
        yield
    except asyncio.CancelledError:
        # Gracefully handle the cancellation that occurs on Ctrl+C shutdown in Windows
        logger.info("Lifespan task cancelled - cleaning up...")
    finally:
        # ---- Shutdown Logic ----
        logger.info("EvasionEye backend shutting down...")
        # Add any explicit cleanup here if needed



# ---- FastAPI App ----
app = FastAPI(
    title="EvasionEye API",
    description="AI-powered illegal number plate detection from traffic footage.",
    version="2.0.0",
    lifespan=lifespan
)

# ---- CORS (allow frontend to call backend) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Health Check ----
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "EvasionEye API", "version": "2.0.0"}

# ---- OCR check logic moved to lifespan warmup ----
# ---- Endpoints ----


# ---- Analyze Video Endpoint ----
@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    """
    Analyze an uploaded traffic video for illegal number plates.

    Pipeline:
      1. Extract every 5th frame from the video
      2. YOLOv26 plate detector -> bounding boxes
      3. Geometric filter -> plate crop -> preprocessing (160x40)
      4. PaddleOCR/EasyOCR/Tesseract -> plate text (confidence >= 0.25)
      5. Indian RTO format validation -> violation detection
      6. Return structured JSON results

    Accepts: MP4, MOV, WEBM video files.
    Returns: JSON with detection results.
    """
    logger.info(f"Received video upload: {video.filename} ({video.content_type})")

    # Validate file type
    allowed_types = {
        'video/mp4', 'video/webm', 'video/quicktime',
        'video/mov', 'video/x-msvideo',
    }
    if video.content_type and video.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {video.content_type}. Use MP4, MOV, or WEBM.",
        )

    temp_path = None
    try:
        # Save uploaded video to temp file
        file_bytes = await video.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty video file.")

        ext = os.path.splitext(video.filename or 'video.mp4')[1] or '.mp4'
        temp_path = save_upload_to_temp(file_bytes, suffix=ext)
        logger.info(f"Saved to temp: {temp_path} ({len(file_bytes)} bytes)")

        # Step 1: Process video frames — every 3rd frame (more sampling = better stabilization)
        logger.info("Starting YOLO + PaddleOCR pipeline...")
        start_time = time.time()
        raw_detections, total_frames, processed_count = process_video(temp_path, frame_interval=5)
        logger.info(f"Raw plate detections from YOLO: {len(raw_detections)}")

        if not raw_detections:
            logger.info("No plate candidates detected by YOLO.")
            return JSONResponse(content={"detections": []})

        # Step 2: Read plate text via OCR + validate format
        # Use frame-level aggregation: group by normalized plate, select highest confidence
        plate_detections = {}  # normalized_plate -> list of detection entries

        logger.info(f"Processing {len(raw_detections)} raw detections...")

        for idx, det in enumerate(raw_detections):
            plate_text, ocr_conf = read_plate(det['raw_crop'], det['crop'])
            if not plate_text:
                logger.debug(f"Detection {idx}: OCR failed or returned empty")
                continue

            # Use regex for normalization
            normalized = re.sub(r'[\s\-]+', '', plate_text).upper()
            
            # --- Apply stabilization ---
            track_id = det.get('track_id')
            if track_id:
                plate_text, ocr_conf = update_tracker_ocr(track_id, plate_text, ocr_conf)
                normalized = re.sub(r'[\s\-]+', '', plate_text).upper()

            if len(normalized) < 3: # Reduced from 6
                logger.debug(f"Detection {idx}: Plate too short after normalization: '{normalized}'")
                continue

            # Validate against Indian RTO rules
            classification = {
                "plate_type": det.get("plate_type"),
                "color": det.get("plate_color"),
                "text_color": det.get("text_color"),
                "hsrp_status": det.get("hsrp_status")
            }
            validation = validate_plate(plate_text, det.get('raw_crop'), classification=classification)

            logger.info(
                f"Detection {idx}: '{validation.detected_plate}' → "
                f"Violation: {validation.violation or 'None'}"
            )

            # Calculate combined confidence (YOLO + OCR) — reuse ocr_conf from read_plate
            yolo_conf = det['confidence']

            # Use detected_plate as fallback if correct_plate is None (legal plates)
            plate_for_format_check = validation.correct_plate or validation.detected_plate
            is_valid_format = bool(PLATE_PATTERN.match(plate_for_format_check))

            combined_conf = calculate_confidence(
                yolo_conf=yolo_conf,
                ocr_conf=ocr_conf,
                is_valid_format=is_valid_format,
                frames_seen=1,  # Individual detection at this stage
                confidence_modifier=validation.confidence_modifier
            )

            # Encode plate crop image as base64 for frontend evidence display
            plate_image_b64 = None
            try:
                if det.get('raw_crop') is not None and det['raw_crop'].size > 0:
                    import base64
                    _, buf = cv2.imencode('.jpg', det['raw_crop'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                    plate_image_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')
            except Exception as enc_err:
                logger.warning(f"Could not encode plate crop: {enc_err}")

            detection_entry = {
                "detected_plate": validation.detected_plate,
                "correct_plate": validation.correct_plate,
                "violation": validation.violation,
                "violations": validation.violations,
                "font_anomaly": validation.font_anomaly,
                "confidence": combined_conf,
                "yolo_conf": yolo_conf,
                "ocr_conf": ocr_conf,
                "confidence_modifier": validation.confidence_modifier,
                "frame": det['frame_number'],
                "plate_image": plate_image_b64,  # base64 JPEG crop for evidence
                "source": "video_analysis",
                "plate_type": validation.plate_type,
                "plate_color": validation.plate_color,
                "hsrp_status": validation.hsrp_status,
            }
            
            # Add vehicle registration info if available
            if validation.vehicle_info:
                detection_entry["vehicle_info"] = validation.vehicle_info


            # Group by normalized plate text for aggregation
            if normalized not in plate_detections:
                plate_detections[normalized] = []
            plate_detections[normalized].append(detection_entry)

        # Step 3: Deduplication + Stabilization
        # Only keep plates seen in >= 2 frames
        MIN_FRAMES_REQUIRED = 2
        
        # Flatten detection lists for deduplicator
        all_flattened_dets = []
        for plate_list in plate_detections.values():
            all_flattened_dets.extend(plate_list)
            
        # Deduplicate using Levenshtein distance
        detections = deduplicate_detections(all_flattened_dets, distance_threshold=2)
        
        # Filter by stabilization requirement and recalculate confidence with stability boost
        final_detections = []
        for det in detections:
            frames_seen = det.get('frames_seen', 1)
            if frames_seen < MIN_FRAMES_REQUIRED:
                continue
                
            # Recalculate confidence with stability boost
            det['confidence'] = calculate_confidence(
                yolo_conf=det.get('yolo_conf', 0.8),
                ocr_conf=det.get('ocr_conf', 0.8),
                is_valid_format=bool(PLATE_PATTERN.match(det.get('correct_plate') or det['detected_plate'])),
                frames_seen=frames_seen,
                confidence_modifier=det.get('confidence_modifier', 1.0)
            )
            
            final_detections.append(det)
            
            # Save the final stabilized + deduplicated detection to DB
            try:
                db.add_detection('CAM-001', det)
            except Exception as db_err:
                logger.warning(f"DB save failed for stabilized video detection: {db_err}")
        
        detections = final_detections

        # Sort by frame number
        detections.sort(key=lambda d: d.get('frame', 0))

        # ---- Final Logging Summary (Phase 9) ----
        process_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Pipeline Execution Summary")
        logger.info(f"  Total frames:      {total_frames}")
        logger.info(f"  Processed frames:  {processed_count}")
        logger.info(f"  Raw detections:    {len(raw_detections)}")
        logger.info(f"  Unique plates:     {len(detections)}")
        logger.info(f"  Violations found:  {len([d for d in detections if d['violation']])}")
        logger.info(f"  Total time:        {process_time:.2f}s")
        logger.info(f"  Avg frame speed:   {process_time/max(1, processed_count):.4f}s/frame")
        logger.info("=" * 60)

        logger.info(
            f"Final results: {len(detections)} unique detections "
            f"(after stabilization + fuzzy dedup), "
            f"{sum(1 for d in detections if d['violation'])} violations"
        )

        return JSONResponse(content={"detections": detections})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(e)}",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass




# ---- Live Monitoring Endpoint ----
@app.get("/api/live-detections")
async def get_live_detections():
    """
    Get live plate detections from a camera feed or video stream.
    
    For demo purposes, this returns mock detections.
    In production, this would connect to a live camera feed.
    """
    # TODO: Connect to actual camera feed
    # For now, return mock detections
    mock_detections = [
        {
            "detected_plate": "WB65D18753",
            "correct_plate": "WB 65 D 18753",
            "violation": None,
            "confidence": 0.95,
            "bbox": [35, 45, 18, 8],  # [x%, y%, width%, height%]
        },
        {
            "detected_plate": "MH12AB1234",
            "correct_plate": "MH 12 AB 1234",
            "violation": None,
            "confidence": 0.92,
            "bbox": [60, 30, 16, 7],
        },
        {
            "detected_plate": "TN1OAB5678",
            "correct_plate": "TN10AB5678",
            "violation": "Character Manipulation",
            "confidence": 0.88,
            "bbox": [25, 55, 17, 8],
        },
    ]
    
    return JSONResponse(content={"detections": mock_detections})


# ---- Camera Management Endpoints ----
@app.get("/api/cameras")
async def get_cameras():
    """Get all cameras"""
    cameras = db.get_all_cameras()
    return JSONResponse(content={"cameras": cameras})


@app.get("/api/cameras/{camera_id}")
async def get_camera_info(camera_id: str):
    """Get camera information"""
    camera = db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return JSONResponse(content={"camera": camera})


class CameraUpdate(BaseModel):
    name: str
    location: str
    resolution: str = "1080p"
    latitude: str = ""
    longitude: str = ""
    address: str = ""
    accuracy: str = ""


@app.put("/api/cameras/{camera_id}")
async def update_camera_info(camera_id: str, data: CameraUpdate):
    """Update camera information"""
    camera = db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    db.update_camera(camera_id, data.dict())
    return JSONResponse(content={"message": "Camera updated successfully"})


# ---- Detection Endpoints ----
@app.get("/api/detections/{camera_id}")
async def get_camera_detections(camera_id: str, violations_only: bool = False, limit: int = 100):
    """Get detections for a camera"""
    detections = db.get_detections(camera_id, limit=limit, violations_only=violations_only)
    
    # Add pretty-printed version for UI
    for det in detections:
        det["pretty_plate"] = format_plate(det.get("correct_plate") or det.get("detected_plate") or "")
        
    return JSONResponse(content={"detections": detections})


@app.get("/api/stats/{camera_id}")
async def get_camera_stats(camera_id: str):
    """Get detection statistics for a camera"""
    stats = db.get_detection_stats(camera_id)
    return JSONResponse(content={"stats": stats})


class Detection(BaseModel):
    detected_plate: str
    correct_plate: Optional[str] = ""
    violation: Optional[str] = None
    violations: Optional[list] = []
    confidence: Optional[float] = 0.0
    yolo_conf: Optional[float] = 0.0
    ocr_conf: Optional[float] = 0.0
    confidence_modifier: Optional[float] = 1.0
    frame: Optional[int] = 0
    frames_seen: Optional[int] = 1
    bbox: Optional[list] = []
    plate_image: Optional[str] = ""
    source: Optional[str] = "live_monitoring"
    vehicle_info: Optional[dict] = {}
    parent_vehicle: Optional[str] = "Unknown"
    vehicle_type: Optional[str] = None
    vehicle_label: Optional[str] = None
    track_id: Optional[Union[int, str]] = None
    status: Optional[str] = None
    timestamp: Optional[str] = None

    class Config:
        extra = "ignore"


@app.post("/api/detections/{camera_id}")
async def add_detection(
    request: Request,
    camera_id: str = Path(..., regex="^[a-zA-Z0-9_-]+$")
):
    """Add a new detection with raw body tracking"""
    raw_body = await request.body()
    try:
        data_str = raw_body.decode('utf-8')
        logger.debug(f"Raw detection payload: {data_str}")
        data = json.loads(data_str)
        detection = Detection(**data)
    except Exception as e:
        logger.error(f"Detection schema validation failed: {e}")
        raise HTTPException(status_code=422, detail=f"Validation failed: {e}")

    try:
        camera = db.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        detection_id = db.add_detection(camera_id, detection.dict())
        logger.info(f"Added detection {detection_id} for camera {camera_id}: {detection.detected_plate}")
        return JSONResponse(content={"id": detection_id, "message": "Detection added successfully"})
    except Exception as e:
        logger.error(f"Error adding detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/detections/{camera_id}")
async def clear_detections(camera_id: str):
    """Clear all detections for a camera"""
    db.clear_all_detections(camera_id)
    return JSONResponse(content={"message": "All detections cleared"})



# Removed local _stabilize_detection, now using plate_stabilizer instance


# ---- Live Camera Stream Processing ----
from collections import deque

detection_buffer = {}  # camera_id -> deque of recent detections
PERSISTENCE_FRAMES = 3

def apply_detection_persistence(cam_id: str, current_detections: list):
    if cam_id not in detection_buffer:
        detection_buffer[cam_id] = deque(maxlen=PERSISTENCE_FRAMES)
        
    buffer = detection_buffer[cam_id]
    
    if len(current_detections) > 0:
        buffer.append({
            "detections": current_detections,
            "timestamp": time.time(),
            "missed_frames": 0
        })
        return current_detections

    if len(buffer) > 0:
        last = buffer[-1]
        age_frames = last.get("missed_frames", 0) + 1
        
        if age_frames <= PERSISTENCE_FRAMES:
            last["missed_frames"] = age_frames
            logger.debug(
                f"[{cam_id}] YOLO missed frame — "
                f"using buffered detection (miss #{age_frames}/{PERSISTENCE_FRAMES})"
            )
            return last["detections"]
            
    return []

@app.post("/api/process-frame")
async def process_camera_frame(
    file: UploadFile = File(...),
    original_width: int = Form(0),
    original_height: int = Form(0),
    camera_id: Optional[str] = Form(default='CAM-001')
):
    """
    Process a single camera frame and return YOLO detections with bounding boxes.
    
    This endpoint is designed for live camera feeds where frames are sent
    one at a time for real-time processing.
    
    Args:
        file: The image frame to process
        original_width: Original webcam width (before downscaling)
        original_height: Original webcam height (before downscaling)
    
    Returns:
        JSON with detected plates and their bounding boxes (as percentages).
    """
    logger.info("Received frame for processing")
    frame_start_time = time.time()
    
    try:
        # Read the uploaded frame
        contents = await file.read()
        logger.info(f"Frame size: {len(contents)} bytes")
        
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode frame")
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Get frame dimensions (processing resolution)
        frame_height, frame_width = frame.shape[:2]
        logger.info(f"Processing dimensions: {frame_width}x{frame_height}")
        
        # Use original dimensions for percentage calculation if provided
        # This ensures bboxes align correctly when displayed on original resolution canvas
        display_width = original_width if original_width > 0 else frame_width
        display_height = original_height if original_height > 0 else frame_height
        
        if original_width > 0 and original_height > 0:
            logger.info(f"Display dimensions: {display_width}x{display_height}")
            # Calculate scaling factors
            scale_x = frame_width / display_width
            scale_y = frame_height / display_height
            logger.info(f"Scaling factors: x={scale_x:.3f}, y={scale_y:.3f}")
        
        # Run YOLO detection
        raw_detections = detect_plates(frame, frame_number=0)
        
        safe_cam_id = camera_id if camera_id else "CAM-001"
        detections = apply_detection_persistence(safe_cam_id, raw_detections)
        
        if len(detections) == 0:
            logger.info("YOLO detected 0 plates")
        else:
            logger.info(f"YOLO detected {len(detections)} plates")
        
        # Convert detections to percentage-based bounding boxes
        results = []
        for idx, det in enumerate(detections):
            bbox = det['bbox']  # [x, y, w, h] in pixels (at processing resolution)
            
            # Scale bbox back to original display resolution if needed
            if original_width > 0 and original_height > 0:
                # Bbox is in processing resolution, scale to display resolution
                display_x = bbox[0] / scale_x
                display_y = bbox[1] / scale_y
                display_w = bbox[2] / scale_x
                display_h = bbox[3] / scale_y
            else:
                # No scaling needed
                display_x = bbox[0]
                display_y = bbox[1]
                display_w = bbox[2]
                display_h = bbox[3]
            
            # Convert to percentages based on display dimensions
            x_pct = (display_x / display_width) * 100
            y_pct = (display_y / display_height) * 100
            w_pct = (display_w / display_width) * 100
            h_pct = (display_h / display_height) * 100
            
            # Convert plate crop to base64 for frontend display
            import base64
            _, buffer = cv2.imencode('.jpg', det['raw_crop'])
            plate_image_base64 = base64.b64encode(buffer).decode('utf-8')
            plate_image_data_url = f"data:image/jpeg;base64,{plate_image_base64}"
            
            # Try to read the plate text
            plate_text, ocr_conf = read_plate(det['raw_crop'], det['crop'])
            
            # Check if OCR detected a decorated plate violation
            ocr_violation = get_last_ocr_violation()
            
            if plate_text:
                # --- Apply stabilization ---
                track_id = det.get('track_id')
                if track_id:
                    plate_text, ocr_conf = update_tracker_ocr(track_id, plate_text, ocr_conf)
                    
                    voted_text = vote_stabilized_plate(str(track_id), plate_text, ocr_conf)
                    if voted_text:
                        plate_text = voted_text


                # Validate the plate using unified grammar_validator
                classification = {
                    "plate_type": det.get("plate_type"),
                    "color": det.get("plate_color"),
                    "text_color": det.get("text_color"),
                    "hsrp_status": det.get("hsrp_status")
                }
                validation = validate_plate(plate_text, det['raw_crop'], classification=classification)
                
                # Override with decorated plate violation if detected
                if ocr_violation and ocr_violation.get("violation") == "RTO_FORMAT_VIOLATION":
                    logger.info(f"Plate '{plate_text}' flagged ILLEGAL — {ocr_violation.get('violation_detail', 'RTO format violation')}")
                    if not validation.violation:
                        validation.violation = "RTO_FORMAT_VIOLATION"
                    if not validation.violations:
                        validation.violations = []
                    validation.violations.append(ocr_violation.get('violation_detail', 'Decorative/Fancy Text Detected'))
                
                logger.info(f"Plate {idx}: '{validation.detected_plate}' - {validation.violation or 'LEGAL'}")
                
                result_entry = {
                    "detected_plate": validation.detected_plate,
                    "correct_plate": validation.correct_plate,
                    "violation": validation.violation,
                    "font_anomaly": validation.font_anomaly,
                    "confidence": min(1.0, round(det['confidence'], 2)),
                    "yolo_conf": det['confidence'],
                    "ocr_conf": ocr_conf,
                    "confidence_modifier": validation.confidence_modifier,
                    "bbox": [
                        round(x_pct, 1),
                        round(y_pct, 1),
                        round(w_pct, 1),
                        round(h_pct, 1)
                    ],
                    "plate_image": plate_image_data_url,
                    "vehicle_info": validation.vehicle_info,
                    "parent_vehicle": det.get("parent_vehicle", "Unknown"),
                    "vehicle_type": det.get("vehicle_type", "Unknown"),
                    "vehicle_label": det.get("vehicle_label", "Unknown"),
                    "violations": validation.violations,
                    "track_id": det.get("track_id"),
                    "plate_type": validation.plate_type,
                    "plate_color": validation.plate_color,
                    "hsrp_status": validation.hsrp_status
                }
                
                # Calculate confidence using new module
                is_valid_format = bool(PLATE_PATTERN.match(validation.correct_plate or validation.detected_plate))
                
                result_entry['confidence'] = calculate_confidence(
                    yolo_conf=det['confidence'],
                    ocr_conf=result_entry['ocr_conf'],
                    is_valid_format=is_valid_format,
                    frames_seen=1,
                    confidence_modifier=validation.confidence_modifier
                )
                
                # ── Stabilization: require 2+ consistent frame readings ──
                # Use a simple regex-based normalization for stabilization key
                normalized = re.sub(r'[\s\-]+', '', validation.detected_plate).upper()
                
                # Deduplication: If this track already saved a plate, ignore other variations
                if det.get("track_id") and plate_stabilizer.is_track_saved(det["track_id"]) and not plate_stabilizer.is_saved(normalized):
                    logger.debug(f"  - Track {det['track_id']} already saved. Skipping variation '{normalized}'")
                    continue

                is_confirmed = plate_stabilizer.stabilize_detection(normalized, result_entry)

                if is_confirmed:
                    # Recalculate with updated frames_seen
                    result_entry['confidence'] = calculate_confidence(
                        yolo_conf=det['confidence'],
                        ocr_conf=result_entry['ocr_conf'],
                        is_valid_format=is_valid_format,
                        frames_seen=result_entry.get('frames_seen', 1),
                        confidence_modifier=validation.confidence_modifier
                    )
                    
                    if not plate_stabilizer.is_saved(normalized):
                        try:
                            db.add_detection(camera_id or 'CAM-001', result_entry)
                            plate_stabilizer.mark_saved(normalized, bbox=result_entry.get('bbox'))
                            logger.info(f"  ✓ Stabilized plate '{normalized}' — saved to DB")
                        except Exception as db_err:
                            logger.warning(f"DB save failed: {db_err}")
                    
                    results.append(result_entry)
                else:
                    # First sighting — show bbox + "Detecting..." until confirmed
                    # Only show if not already saved for this track
                    if not (det.get("track_id") and plate_stabilizer.is_track_saved(det["track_id"])):
                        results.append({
                            "detected_plate": "",
                            "correct_plate": None,
                            "violation": None,
                            "status": "DETECTING",
                            "confidence": result_entry['confidence'],
                            "bbox": result_entry['bbox'],
                            "track_id": det.get("track_id")
                        })
            elif det['confidence'] > 0.3:
                # YOLO detected something but OCR failed — still show the box
                results.append({
                    "detected_plate": "",
                    "correct_plate": None,
                    "violation": None,
                    "status": "DETECTING",
                    "confidence": det['confidence'],
                    "bbox": [
                        round(x_pct, 1),
                        round(y_pct, 1),
                        round(w_pct, 1),
                        round(h_pct, 1)
                    ],
                    "track_id": det.get("track_id")
                })
        
        proc_time = (time.time() - frame_start_time) * 1000
        logger.info(f"Frame processed in {proc_time:.1f}ms")
        
        return JSONResponse(content={
            "detections": results,
            "latency_ms": round(proc_time, 1)
        })
        
    except Exception as e:
        logger.error(f"Error processing live frame: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---- Root Redirect to Frontend ----
@app.get("/", include_in_schema=False)
async def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/index.html")

# ---- Serve frontend static files (MOUNTED LAST) ----
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..')
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
