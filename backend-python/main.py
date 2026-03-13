"""
RoadVision — FastAPI Backend
Main application entry point.

Provides the /analyze-video endpoint that processes uploaded traffic videos,
detects number plates using YOLO plate detector + CRNN recognizer, validates
plate formats against Indian RTO rules, and returns structured detection results.

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


# ---- Startup: check model files ----
@app.on_event("startup")
async def startup_check():
    """Log model file status on server startup."""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    detector_path = os.path.join(models_dir, 'license_plate_detector.pt')
    crnn_path = os.path.join(models_dir, 'crnn.pth')

    logger.info("=" * 60)
    logger.info("RoadVision Backend v2.0 — Startup Check")
    logger.info("=" * 60)

    if os.path.exists(detector_path):
        logger.info(f"  ✓ Plate detector:  {detector_path}")
    else:
        logger.error(f"  ✗ Plate detector MISSING: {detector_path}")
        logger.error("    Will fall back to yolov8s.pt (auto-downloaded by Ultralytics).")
        logger.error("    Place 'license_plate_detector.pt' in backend-python/models/ for best results.")

    if os.path.exists(crnn_path):
        logger.info(f"  ✓ CRNN weights:    {crnn_path}")
    else:
        logger.warning(f"  ⚠ CRNN weights MISSING: {crnn_path}")
        logger.warning("    Falling back to PaddleOCR for plate text recognition.")
        logger.warning("    Place 'crnn.pth' in backend-python/models/ to use CRNN instead.")

    # Check PaddleOCR availability
    try:
        import paddleocr
        logger.info("  ✓ PaddleOCR:       Available (fallback OCR engine)")
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

    Pipeline: YOLO plate detector → geometric filter → plate crop
    preprocessing → CRNN recognition → text normalization → Indian
    plate validation.

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

        # Step 1: Process video frames — YOLO plate detection + preprocessing
        logger.info("Starting YOLO + CRNN pipeline...")
        raw_detections = process_video(temp_path, frame_interval=20)
        logger.info(f"Raw plate detections: {len(raw_detections)}")

        if not raw_detections:
            logger.info("No plate candidates detected.")
            return JSONResponse(content={"detections": []})

        # Step 2: Read plate text via CRNN + validate format
        detections: List[dict] = []
        seen_plates = set()

        for det in raw_detections:
            # Read plate text using CRNN
            plate_text = read_plate(det['crop'])
            if not plate_text:
                continue

            normalized = normalize_plate(plate_text)
            if len(normalized) < 6:
                continue

            # Deduplicate
            if normalized in seen_plates:
                continue
            seen_plates.add(normalized)

            # Validate against Indian RTO rules
            validation = validate_plate(plate_text)

            # Calculate combined confidence (YOLO + CRNN)
            yolo_conf = det['confidence']
            crnn_conf = get_read_confidence(det['crop'])
            combined_conf = round(
                (yolo_conf * 0.4 + crnn_conf * 0.6) * validation.confidence_modifier,
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
        "message": "RoadVision API v2.0 — YOLO + CRNN Pipeline",
        "docs": "/docs",
        "health": "/health",
        "analyze": "POST /analyze-video",
    }
