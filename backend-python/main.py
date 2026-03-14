"""
RoadVision — FastAPI Backend
Main application entry point.

Provides the /analyze-video endpoint that processes uploaded traffic videos,
detects number plates using a dedicated YOLOv8 plate detector + PaddleOCR,
validates plate formats against Indian RTO rules, and returns structured
detection results.

Run with: uvicorn main:app --reload
API docs: http://localhost:8000/docs
"""

import logging
import os
import tempfile
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from processing.video_processor import process_video, save_upload_to_temp
from recognition.plate_reader import read_plate, get_read_confidence, load_plate_model
from rules.plate_rules import validate_plate, normalize_plate
import database as db


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("roadvision")

# ---- FastAPI App ----
app = FastAPI(
    title="RoadVision API",
    description="AI-powered illegal number plate detection from traffic footage.",
    version="2.0.0",
)

# ---- CORS (allow frontend to call backend) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Serve frontend static files ----
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..')


# ---- Health Check ----
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "RoadVision API", "version": "2.0.0"}


# ---- Startup: load model + check dependencies ----
@app.on_event("startup")
async def startup_check():
    """Load the YOLO plate detector at startup and verify dependencies."""
    # Initialize database
    db.init_database()
    logger.info("Database initialized")
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    detector_path = os.path.join(models_dir, 'license_plate_detector.pt')
    crnn_path = os.path.join(models_dir, 'crnn.pth')

    logger.info("=" * 60)
    logger.info("RoadVision Backend v2.0 — Startup")
    logger.info("=" * 60)

    # ---- Load YOLO plate detector (REQUIRED) ----
    try:
        load_plate_model()
        logger.info(f"  ✓ Plate detector:  {detector_path}")
    except FileNotFoundError as e:
        logger.error(f"  ✗ FATAL: {e}")
        logger.error("=" * 60)
        raise RuntimeError(
            f"License plate detector model not found at: {detector_path}\n"
            f"Place 'license_plate_detector.pt' (from keremberke/yolov8-license-plate) "
            f"in backend-python/models/ and restart the server."
        )

    # ---- Check OCR availability ----
    ocr_found = False
    
    if os.path.exists(crnn_path):
        logger.info(f"  ✓ CRNN weights:    {crnn_path}")
        ocr_found = True
    else:
        logger.warning(f"  ⚠ CRNN weights MISSING: {crnn_path}")

    # Check PaddleOCR
    try:
        import paddleocr
        logger.info("  ✓ PaddleOCR:       Available")
        ocr_found = True
    except ImportError:
        logger.debug("  - PaddleOCR:       Not installed")

    # Check EasyOCR
    try:
        import easyocr
        logger.info("  ✓ EasyOCR:         Available")
        ocr_found = True
    except ImportError:
        logger.debug("  - EasyOCR:         Not installed")

    # Check Tesseract
    try:
        import pytesseract
        from recognition.crnn_recognizer import _tesseract_recognize # Trigger path resolution
        # We don't call it, just check if it's configured
        tess_path = pytesseract.pytesseract.tesseract_cmd
        if os.path.exists(tess_path) or os.system("tesseract --version > nul 2>&1") == 0:
            logger.info(f"  ✓ Tesseract:       Available ({tess_path if os.path.exists(tess_path) else 'in PATH'})")
            ocr_found = True
        else:
            logger.warning("  ⚠ Tesseract:       Executable not found")
    except ImportError:
        logger.debug("  - Tesseract:       Not installed")

    if not ocr_found:
        logger.error("  ✗ FATAL: No OCR engine available (CRNN, PaddleOCR, EasyOCR, or Tesseract)!")
        logger.error("    Please install at least one fallback: pip install easyocr")
    else:
        logger.info("  ✓ OCR Pipeline:    Ready (with fallback support)")

    logger.info("=" * 60)
    logger.info("Ready at: http://localhost:8000")
    logger.info("API docs: http://localhost:8000/docs")
    logger.info("=" * 60)


# ---- Analyze Video Endpoint ----
@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    """
    Analyze an uploaded traffic video for illegal number plates.

    Pipeline:
      1. Extract every 5th frame from the video
      2. YOLOv8 plate detector → bounding boxes
      3. Geometric filter → plate crop → preprocessing (160×40)
      4. PaddleOCR/CRNN → plate text (confidence ≥ 0.6)
      5. Indian RTO format validation → violation detection
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

        # Step 1: Process video frames — every 5th frame
        logger.info("Starting YOLO + PaddleOCR pipeline...")
        raw_detections = process_video(temp_path, frame_interval=5)
        logger.info(f"Raw plate detections from YOLO: {len(raw_detections)}")

        if not raw_detections:
            logger.info("No plate candidates detected by YOLO.")
            return JSONResponse(content={"detections": []})

        # Step 2: Read plate text via OCR + validate format
        # Use frame-level aggregation: group by normalized plate, select highest confidence
        plate_detections = {}  # normalized_plate -> list of detection entries

        logger.info(f"Processing {len(raw_detections)} raw detections...")

        for idx, det in enumerate(raw_detections):
            # Read plate text using PaddleOCR/CRNN/Tesseract - pass both raw and preprocessed
            plate_text = read_plate(det['raw_crop'], det['crop'])
            if not plate_text:
                logger.debug(f"Detection {idx}: OCR failed or returned empty")
                continue

            normalized = normalize_plate(plate_text)
            if len(normalized) < 6:
                logger.debug(f"Detection {idx}: Plate too short after normalization: '{normalized}'")
                continue

            # Validate against Indian RTO rules
            validation = validate_plate(plate_text, det.get('raw_crop'))

            logger.info(
                f"Detection {idx}: '{validation.detected_plate}' → "
                f"Violation: {validation.violation or 'None'}"
            )

            # Calculate combined confidence (YOLO + OCR) - pass both images
            yolo_conf = det['confidence']
            ocr_conf = get_read_confidence(det['raw_crop'], det['crop'])
            
            # Base combined confidence with weighted average (YOLO: 0.4, OCR: 0.6)
            base_conf = (yolo_conf * 0.4 + ocr_conf * 0.6)
            
            # Apply confidence boost for plates matching Indian format
            # Valid format plates (matching PLATE_PATTERN) get a boost to reach >= 0.50 threshold
            # This applies even to unregistered vehicles, as long as the format is correct
            format_boost = 1.0
            from rules.plate_rules import PLATE_PATTERN
            # Use detected_plate as fallback if correct_plate is None (legal plates)
            plate_for_format_check = validation.correct_plate or validation.detected_plate
            if PLATE_PATTERN.match(plate_for_format_check):
                # Matches Indian RTO format - apply boost
                format_boost = 1.15
            
            # Calculate combined confidence and cap at 1.0
            combined_conf = min(
                1.0,
                round(
                    base_conf * format_boost * validation.confidence_modifier,
                    2,
                )
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
                "font_anomaly": validation.font_anomaly,
                "confidence": combined_conf,
                "frame": det['frame_number'],
                "bbox": det['bbox'],  # [x, y, width, height]
                "plate_image": plate_image_b64,  # base64 JPEG crop for evidence
                "source": "video_analysis",
            }
            
            # Add vehicle registration info if available
            if validation.vehicle_info:
                detection_entry["vehicle_info"] = validation.vehicle_info


            # Group by normalized plate text for aggregation
            if normalized not in plate_detections:
                plate_detections[normalized] = []
            plate_detections[normalized].append(detection_entry)

        # Step 3: Frame-level aggregation with fuzzy matching
        # Group similar plates together using Levenshtein distance
        detections: List[dict] = []
        processed_plates = set()
        
        for normalized_plate, detection_list in plate_detections.items():
            # Check if this plate is similar to any already processed plate
            is_duplicate = False
            for processed in processed_plates:
                # Calculate Levenshtein distance
                distance = _levenshtein_distance(normalized_plate, processed)
                # If plates are similar (≤3 char difference), consider it a duplicate
                # This catches OCR variants like TN57ADJ80 vs TN57AD3604
                if distance <= 3 and abs(len(normalized_plate) - len(processed)) <= 3:
                    logger.debug(f"Skipping duplicate: '{normalized_plate}' ≈ '{processed}' (distance: {distance})")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Select the detection with highest confidence
                best_detection = max(detection_list, key=lambda d: d['confidence'])
                detections.append(best_detection)
                processed_plates.add(normalized_plate)

                # Save the final deduplicated detection to DB
                try:
                    db.add_detection('CAM-001', best_detection)
                except Exception as db_err:
                    logger.warning(f"DB save failed for deduplicated video detection: {db_err}")
                
                logger.debug(
                    f"Plate '{normalized_plate}': aggregated {len(detection_list)} detections, "
                    f"selected frame {best_detection['frame']} with confidence {best_detection['confidence']}"
                )

        # Sort by frame number
        detections.sort(key=lambda d: d.get('frame', 0))

        logger.info(
            f"Final results: {len(detections)} unique detections "
            f"(after fuzzy deduplication), "
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
    return JSONResponse(content={"detections": detections})


@app.get("/api/stats/{camera_id}")
async def get_camera_stats(camera_id: str):
    """Get detection statistics for a camera"""
    stats = db.get_detection_stats(camera_id)
    return JSONResponse(content={"stats": stats})


class Detection(BaseModel):
    detected_plate: str
    correct_plate: str = ""
    violation: Optional[str] = None
    confidence: float
    frame: int = 0
    bbox: list = []
    plate_image: str = ""
    source: str = "live_monitoring"
    vehicle_info: dict = {}


@app.post("/api/detections/{camera_id}")
async def add_detection(camera_id: str, detection: Detection):
    """Add a new detection"""
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



# ---- Stabilization Tracker (from RoadVision Deep Dive) ----
# "Do not log a plate the first millisecond you see it. Wait for at least
# 2-3 frames of consistent readings to ensure the OCR has settled."
import time as _time

_plate_tracker: dict = {}  # normalized_plate -> {count, best_conf, best_entry, last_seen, saved}
STABILIZATION_FRAMES = 2   # Require N consistent frame readings before confirming
TRACKER_EXPIRY_SEC = 30    # Expire stale tracker entries after 30s


def _stabilize_detection(normalized_plate: str, result_entry: dict, camera_id: str) -> bool:
    """
    Track a plate across frames. Returns True when the plate is confirmed
    (seen in ≥ STABILIZATION_FRAMES frames) and should be sent to client.
    Saves to DB only on first confirmation.
    """
    global _plate_tracker
    now = _time.time()

    # Expire stale entries
    stale = [k for k, v in _plate_tracker.items() if now - v['last_seen'] > TRACKER_EXPIRY_SEC]
    for k in stale:
        del _plate_tracker[k]

    if normalized_plate in _plate_tracker:
        entry = _plate_tracker[normalized_plate]
        entry['count'] += 1
        entry['last_seen'] = now
        # Keep highest confidence reading
        if result_entry.get('confidence', 0) > entry['best_conf']:
            entry['best_conf'] = result_entry.get('confidence', 0)
            entry['best_entry'] = result_entry

        if entry['count'] >= STABILIZATION_FRAMES and not entry['saved']:
            # Plate confirmed — save to DB once
            entry['saved'] = True
            try:
                db.add_detection(camera_id or 'CAM-001', entry['best_entry'])
                logger.info(f"  ✓ Stabilized plate '{normalized_plate}' after {entry['count']} frames — saved to DB")
            except Exception as db_err:
                logger.warning(f"DB save failed for stabilized detection: {db_err}")
            return True
        elif entry['count'] >= STABILIZATION_FRAMES:
            return True  # Already saved, still show on frontend
        else:
            return False  # Not yet confirmed
    else:
        _plate_tracker[normalized_plate] = {
            'count': 1,
            'best_conf': result_entry.get('confidence', 0),
            'best_entry': result_entry,
            'last_seen': now,
            'saved': False,
        }
        return False  # First sighting — don't show yet


# ---- Live Camera Stream Processing ----
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
        from recognition.plate_reader import detect_plates
        detections = detect_plates(frame, frame_number=0)
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
            from recognition.plate_reader import read_plate
            plate_text = read_plate(det['raw_crop'], det['crop'])
            
            if plate_text:
                # Validate the plate
                from rules.plate_rules import validate_plate, normalize_plate as _norm_plate
                validation = validate_plate(plate_text, det['raw_crop'])
                
                logger.info(f"Plate {idx}: '{validation.detected_plate}' - {validation.violation or 'LEGAL'}")
                
                result_entry = {
                    "detected_plate": validation.detected_plate,
                    "correct_plate": validation.correct_plate,
                    "violation": validation.violation,
                    "font_anomaly": validation.font_anomaly,
                    "confidence": min(1.0, round(det['confidence'], 2)),
                    "bbox": [
                        round(x_pct, 1),
                        round(y_pct, 1),
                        round(w_pct, 1),
                        round(h_pct, 1)
                    ],
                    "plate_image": plate_image_data_url
                }
                
                # Add vehicle registration info if available
                if validation.vehicle_info:
                    result_entry["vehicle_info"] = validation.vehicle_info

                # ── Stabilization: require 2+ consistent frame readings ──
                normalized = _norm_plate(validation.detected_plate)
                is_confirmed = _stabilize_detection(normalized, result_entry, camera_id)

                if is_confirmed:
                    # Plate confirmed across multiple frames — show with full data
                    results.append(result_entry)
                else:
                    # First sighting — show bbox + "Detecting..." until confirmed
                    results.append({
                        "detected_plate": "Detecting...",
                        "correct_plate": None,
                        "violation": None,
                        "confidence": min(1.0, round(det['confidence'], 2)),
                        "bbox": [
                            round(x_pct, 1),
                            round(y_pct, 1),
                            round(w_pct, 1),
                            round(h_pct, 1)
                        ],
                        "plate_image": plate_image_data_url
                    })
            else:
                # No OCR text, just show the detection
                logger.info(f"Plate {idx}: OCR failed, showing as 'Detecting...'")
                results.append({
                    "detected_plate": "Detecting...",
                    "correct_plate": "",
                    "violation": None,
                    "confidence": min(1.0, round(det['confidence'], 2)),
                    "bbox": [
                        round(x_pct, 1),
                        round(y_pct, 1),
                        round(w_pct, 1),
                        round(h_pct, 1)
                    ],
                    "plate_image": plate_image_data_url
                })
        
        logger.info(f"Returning {len(results)} results to frontend")
        return JSONResponse(content={"detections": results})
        
    except Exception as e:
        logger.error(f"Frame processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ---- Mount Frontend Application (Must be last) ----
if os.path.exists(os.path.join(FRONTEND_DIR, 'index.html')):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
