# RoadVision 🚗

AI-powered license plate detection and validation system for traffic enforcement.

## Overview

RoadVision analyzes traffic camera footage to detect and validate vehicle license plates, identifying potential violations such as:

- Character manipulation (O→0, I→1, B→8, S→5)
- Spacing manipulation
- Tampered plates
- Invalid state codes
- Pattern mismatches

## Features

- **Video Analysis:** Upload traffic footage for batch processing
- **Live Monitoring:** Real-time camera feed analysis
- **Detection History:** Review past detections and violations
- **Evidence Storage:** Capture and store violation evidence
- **Indian RTO Validation:** Validates against Indian number plate format (AA NN AA NNNN)

## Tech Stack

**Frontend:**
- HTML5, CSS3, JavaScript
- Material Symbols icons
- LocalStorage for evidence management

**Backend:**
- Python 3.8+
- FastAPI (REST API)
- YOLOv26 (plate detection)
- PaddleOCR/EasyOCR/Tesseract ensemble (text recognition)
- OpenCV (video processing)

## Quick Start

```bash
# 1. Install dependencies
cd RoadVision/backend-python
pip install -r requirements.txt

# 2. Start backend
uvicorn main:app --reload --port 8000

# 3. Start frontend (new terminal)
cd RoadVision
python -m http.server 3000

# 4. Open browser
# http://localhost:3000
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze-video` | POST | Upload video for analysis |
| `/api/process-frame` | POST | Process single camera frame |
| `/api/live-detections` | GET | Get live detection stream |
| `/docs` | GET | Interactive API documentation |

## Detection Pipeline

```
Video Upload
    ↓
Frame Extraction (every 5th frame)
    ↓
YOLOv26 Plate Detection
    ↓
Geometric Filtering (aspect ratio, size, position)
    ↓
Image Preprocessing (CLAHE, adaptive threshold)
    ↓
OCR Recognition (PaddleOCR → EasyOCR → Tesseract ensemble)
    ↓
Indian RTO Format Validation
    ↓
Violation Detection
    ↓
JSON Response
```

## Configuration

Edit `config.js` to customize:

```javascript
const RoadVisionConfig = {
  API_BASE_URL: 'http://localhost:8000',  // Backend URL
  MAX_VIDEO_DURATION: 60,                  // Max video length (seconds)
  FRAME_INTERVAL: 5,                       // Process every Nth frame
  DETECTION_CONFIDENCE_THRESHOLD: 0.6      // Min confidence score
};
```

## Project Structure

```
RoadVision/
├── index.html                 # Main dashboard
├── monitoring.html            # Live monitoring
├── history.html              # Detection history
├── config.js                 # Configuration
├── styles.css                # Styling
├── components/               # UI components
│   ├── sidebar.html
│   ├── header.html
│   ├── evidence-store.js
│   └── ...
├── backend-python/
│   ├── main.py              # FastAPI server
│   ├── requirements.txt     # Dependencies
│   ├── models/              # AI models
│   │   ├── license_plate_detector.pt
│   │   └── yolov26n.pt
│   ├── processing/          # Video processing
│   │   └── video_processor.py
│   ├── recognition/         # OCR & detection
│   │   ├── plate_reader.py
│   │   └── (OCR engines)
│   └── rules/               # Validation rules
│       └── plate_rules.py
└── docs/
    ├── SETUP_GUIDE.md       # Setup instructions
    └── ANALYSIS_AND_FIXES.md # Technical analysis
```

## Validation Rules

RoadVision validates plates against Indian RTO format:

**Format:** `AA NN AA NNNN`
- AA: State code (2 letters)
- NN: District code (2 digits)
- AA: Series letters (1-3 letters)
- NNNN: Registration number (1-4 digits)

**Supported State Codes:**
AN, AP, AR, AS, BR, CG, CH, DD, DL, GA, GJ, HP, HR, JH, JK, KA, KL, LA, LD, MH, ML, MN, MP, MZ, NL, OD, PB, PY, RJ, SK, TN, TR, TS, UK, UP, WB

## Performance

- **Frame Processing:** ~200ms per frame (CPU)
- **Video Analysis:** ~30 seconds for 60-second video
- **Detection Accuracy:** 85-95% (depends on video quality)
- **OCR Accuracy:** 80-90% (with fallback chain)

## Troubleshooting

See [SETUP_GUIDE.md](SETUP_GUIDE.md#troubleshooting) for common issues and solutions.

## License

This project is for educational and research purposes.

## Credits

- YOLOv8: Ultralytics
- License Plate Detector: keremberke/yolov8-license-plate
- OCR Engines: EasyOCR, PaddleOCR, Tesseract

---

**Version:** 2.0.0  
**Last Updated:** March 14, 2026
