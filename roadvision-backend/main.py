"""
RoadVision — FastAPI Backend
Main application entry point.

Provides the /analyze-video endpoint that processes uploaded traffic videos,
detects number plates using YOLO + EasyOCR, validates plate formats against
Indian RTO rules, and returns structured detection results.

Run with: uvicorn main:app --reload
API docs: http://localhost:8000/docs
"""

import logging
import os
import tempfile
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from processing.video_processor import process_video, save_upload_to_temp
from recognition.plate_reader import read_plate, get_read_confidence
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
    version="1.0.0",
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
# Mount the parent RoadVision directory as static files so the
# frontend is accessible at http://localhost:8000/
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..')
if os.path.exists(os.path.join(FRONTEND_DIR, 'index.html')):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ---- Health Check ----
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "RoadVision API"}


# ---- Analyze Video Endpoint ----
@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    """
    Analyze an uploaded traffic video for illegal number plates.

    Accepts: MP4, MOV, WEBM video files (max 1 minute).
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

        # Determine file extension from filename
        ext = os.path.splitext(video.filename or 'video.mp4')[1] or '.mp4'
        temp_path = save_upload_to_temp(file_bytes, suffix=ext)
        logger.info(f"Saved to temp: {temp_path} ({len(file_bytes)} bytes)")

        # Step 1: Process video frames with YOLO detection
        logger.info("Starting video processing pipeline...")
        raw_detections = process_video(temp_path, frame_interval=20)
        logger.info(f"Raw detections from YOLO: {len(raw_detections)}")

        if not raw_detections:
            logger.info("No vehicle/plate candidates detected.")
            return JSONResponse(content={"detections": []})

        # Step 2: Read plate text + validate format for each detection
        detections: List[dict] = []
        seen_plates = set()  # Deduplicate same plate across frames

        for det in raw_detections:
            # Read plate text using EasyOCR
            plate_text = read_plate(det['crop'])
            if not plate_text:
                continue

            normalized = normalize_plate(plate_text)
            if len(normalized) < 4:
                continue

            # Deduplicate: skip if we've already seen this plate
            if normalized in seen_plates:
                continue
            seen_plates.add(normalized)

            # Validate against Indian RTO rules
            validation = validate_plate(plate_text)

            # Calculate combined confidence
            yolo_conf = det['confidence']
            ocr_conf = get_read_confidence(det['crop'])
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
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# ---- Root redirect ----
@app.get("/")
async def root():
    return {
        "message": "RoadVision API",
        "docs": "/docs",
        "health": "/health",
        "analyze": "POST /analyze-video",
    }
