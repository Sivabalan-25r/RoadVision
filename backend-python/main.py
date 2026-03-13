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
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from processing.video_processor import process_video, save_upload_to_temp
from recognition.plate_reader import read_plate, get_read_confidence, load_plate_model
from rules.plate_rules import validate_plate, normalize_plate

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
if os.path.exists(os.path.join(FRONTEND_DIR, 'index.html')):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ---- Health Check ----
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "RoadVision API", "version": "2.0.0"}


# ---- Startup: load model + check dependencies ----
@app.on_event("startup")
async def startup_check():
    """Load the YOLO plate detector at startup and verify dependencies."""
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
    if os.path.exists(crnn_path):
        logger.info(f"  ✓ CRNN weights:    {crnn_path}")
    else:
        logger.warning(f"  ⚠ CRNN weights MISSING: {crnn_path}")
        logger.warning("    Will use PaddleOCR as OCR engine.")

    try:
        import paddleocr
        logger.info("  ✓ PaddleOCR:       Available")
    except ImportError:
        if not os.path.exists(crnn_path):
            logger.error("  ✗ PaddleOCR NOT INSTALLED and CRNN weights missing!")
            logger.error("    Run: pip install paddlepaddle paddleocr")

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
        detections: List[dict] = []
        seen_plates = set()

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

            # Deduplicate
            if normalized in seen_plates:
                logger.debug(f"Detection {idx}: Duplicate plate '{normalized}'")
                continue
            seen_plates.add(normalized)

            # Validate against Indian RTO rules
            validation = validate_plate(plate_text)

            logger.info(
                f"Detection {idx}: '{validation.detected_plate}' → "
                f"Violation: {validation.violation or 'None'}"
            )

            # Calculate combined confidence (YOLO + OCR) - pass both images
            yolo_conf = det['confidence']
            ocr_conf = get_read_confidence(det['raw_crop'], det['crop'])
            combined_conf = round(
                (yolo_conf * 0.4 + ocr_conf * 0.6) * validation.confidence_modifier,
                2,
            )

            detection_entry = {
                "detected_plate": validation.detected_plate,
                "correct_plate": validation.correct_plate,
                "violation": validation.violation,
                "confidence": combined_conf,
                "frame": det['frame_number'],
                "bbox": det['bbox'],  # [x, y, width, height]
            }

            detections.append(detection_entry)

        # Sort by frame number
        detections.sort(key=lambda d: d.get('frame', 0))

        logger.info(
            f"Final results: {len(detections)} detections "
            f"({sum(1 for d in detections if d['violation'])} violations)"
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


# ---- Root redirect ----
@app.get("/")
async def root():
    return {
        "message": "RoadVision API v2.0 — YOLOv8 + PaddleOCR Pipeline",
        "docs": "/docs",
        "health": "/health",
        "analyze": "POST /analyze-video",
        "live": "GET /api/live-detections",
    }


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


# ---- Live Camera Stream Processing ----
@app.post("/api/process-frame")
async def process_camera_frame(file: UploadFile = File(...)):
    """
    Process a single camera frame and return YOLO detections with bounding boxes.
    
    This endpoint is designed for live camera feeds where frames are sent
    one at a time for real-time processing.
    
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
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        logger.info(f"Frame dimensions: {frame_width}x{frame_height}")
        
        # Run YOLO detection
        from recognition.plate_reader import detect_plates
        detections = detect_plates(frame, frame_number=0)
        logger.info(f"YOLO detected {len(detections)} plates")
        
        # Convert detections to percentage-based bounding boxes
        results = []
        for idx, det in enumerate(detections):
            bbox = det['bbox']  # [x, y, w, h] in pixels
            
            # Convert to percentages
            x_pct = (bbox[0] / frame_width) * 100
            y_pct = (bbox[1] / frame_height) * 100
            w_pct = (bbox[2] / frame_width) * 100
            h_pct = (bbox[3] / frame_height) * 100
            
            # Try to read the plate text
            from recognition.plate_reader import read_plate
            plate_text = read_plate(det['raw_crop'], det['crop'])
            
            if plate_text:
                # Validate the plate
                from rules.plate_rules import validate_plate
                validation = validate_plate(plate_text)
                
                logger.info(f"Plate {idx}: '{validation.detected_plate}' - {validation.violation or 'LEGAL'}")
                
                results.append({
                    "detected_plate": validation.detected_plate,
                    "correct_plate": validation.correct_plate,
                    "violation": validation.violation,
                    "confidence": round(det['confidence'], 2),
                    "bbox": [
                        round(x_pct, 1),
                        round(y_pct, 1),
                        round(w_pct, 1),
                        round(h_pct, 1)
                    ]
                })
            else:
                # No OCR text, just show the detection
                logger.info(f"Plate {idx}: OCR failed, showing as 'Detecting...'")
                results.append({
                    "detected_plate": "Detecting...",
                    "correct_plate": "",
                    "violation": None,
                    "confidence": round(det['confidence'], 2),
                    "bbox": [
                        round(x_pct, 1),
                        round(y_pct, 1),
                        round(w_pct, 1),
                        round(h_pct, 1)
                    ]
                })
        
        logger.info(f"Returning {len(results)} results to frontend")
        return JSONResponse(content={"detections": results})
        
    except Exception as e:
        logger.error(f"Frame processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
