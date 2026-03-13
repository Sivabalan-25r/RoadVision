# RoadVision Setup Guide

Complete guide to set up and run the RoadVision application.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- A modern web browser (Chrome, Firefox, Edge, Safari)

## Quick Start

### 1. Install Python Dependencies

```bash
cd RoadVision/backend-python
pip install -r requirements.txt
```

This will install:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- OpenCV (video processing)
- PyTorch (deep learning)
- Ultralytics YOLOv8 (object detection)
- EasyOCR, PaddleOCR, Tesseract (text recognition)

### 2. Verify Model Files

Check that the required model files are present:

```bash
ls models/
```

You should see:
- `license_plate_detector.pt` ✓ (YOLO plate detector - REQUIRED)
- `yolov8n.pt` ✓ (General YOLO model)
- `crnn.pth` (Optional - CRNN text recognizer)

If `license_plate_detector.pt` is missing, the server will not start.

### 3. Start the Backend Server

```bash
cd RoadVision/backend-python
uvicorn main:app --reload --port 8000
```

You should see:
```
✓ Plate detector:  models/license_plate_detector.pt
✓ OCR Pipeline:    Ready (with fallback support)
Ready at: http://localhost:8000
API docs: http://localhost:8000/docs
```

### 4. Start the Frontend

Open a new terminal:

```bash
cd RoadVision
python -m http.server 3000
```

Or simply open `RoadVision/index.html` directly in your browser.

### 5. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Configuration

### API Endpoint Configuration

The frontend automatically detects whether it's running locally or in production:

- **Local Development:** Uses `http://localhost:8000`
- **Production:** Uses the same origin as the frontend

To customize, edit `RoadVision/config.js`:

```javascript
const RoadVisionConfig = {
  API_BASE_URL: 'https://your-backend-url.com',
  // ... other settings
};
```

### Backend Configuration

Edit `RoadVision/backend-python/main.py` to configure:

- **CORS origins** (line 47): Change `allow_origins=["*"]` for production
- **Frame interval** (line 177): Adjust `frame_interval=5` for performance
- **Confidence thresholds**: See `recognition/plate_reader.py`

## Troubleshooting

### Backend won't start

**Error:** `License plate detector model not found`
- **Solution:** Ensure `license_plate_detector.pt` is in `backend-python/models/`

**Error:** `No OCR engine available`
- **Solution:** Install at least one OCR engine:
  ```bash
  pip install easyocr
  # or
  pip install paddlepaddle paddleocr
  ```

### Frontend can't connect to backend

**Error:** `Failed to fetch` or CORS errors
- **Solution:** 
  1. Verify backend is running: http://localhost:8000/health
  2. Check browser console for errors
  3. Ensure CORS is enabled in `main.py`

### Video upload fails

**Error:** `Unsupported video format`
- **Solution:** Use MP4, MOV, or WEBM format

**Error:** `Video too long`
- **Solution:** Trim video to under 60 seconds

### Poor detection accuracy

- Ensure video quality is good (720p or higher recommended)
- Check that plates are clearly visible
- Verify OCR engines are installed correctly
- Review logs in terminal for processing details

## Production Deployment

### Backend

1. Update CORS origins in `main.py`:
   ```python
   allow_origins=["https://your-frontend-domain.com"]
   ```

2. Use a production ASGI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

3. Set up reverse proxy (nginx/Apache) with HTTPS

### Frontend

1. Update `config.js` with production API URL
2. Serve via web server (nginx, Apache, or FastAPI static mount)
3. Enable HTTPS
4. Configure proper caching headers

## Testing

### Test Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "service": "RoadVision API", "version": "2.0.0"}
```

### Test Video Analysis

```bash
curl -X POST http://localhost:8000/analyze-video \
  -F "video=@test_dashcam.mp4"
```

### Test Frontend

1. Open http://localhost:3000
2. Upload a test video
3. Click "Analyze Video"
4. Verify detections appear

## Performance Tips

1. **Frame Interval:** Increase to 10 for faster processing (less accuracy)
2. **Video Resolution:** Downscale large videos before upload
3. **GPU Acceleration:** Install CUDA-enabled PyTorch for faster inference
4. **Caching:** Enable browser caching for static assets

## Support

For issues or questions:
1. Check the logs in the terminal
2. Review `ANALYSIS_AND_FIXES.md` for known issues
3. Visit the API docs at http://localhost:8000/docs
4. Check browser console for frontend errors

## Architecture

```
Frontend (HTML/JS)
    ↓ HTTP POST
Backend (FastAPI)
    ↓
Video Processor → Frame Extraction
    ↓
YOLO Detector → Plate Detection
    ↓
OCR Engine → Text Recognition
    ↓
Validation Rules → Violation Detection
    ↓
JSON Response → Frontend Display
```

## File Structure

```
RoadVision/
├── index.html              # Main dashboard
├── monitoring.html         # Live monitoring
├── history.html           # Detection history
├── config.js              # Configuration (NEW)
├── styles.css             # Styling
├── components/            # Reusable UI components
└── backend-python/
    ├── main.py            # FastAPI server
    ├── requirements.txt   # Python dependencies
    ├── models/            # AI model files
    ├── processing/        # Video processing
    ├── recognition/       # OCR and detection
    └── rules/             # Validation rules
```

---

**Last Updated:** March 14, 2026
