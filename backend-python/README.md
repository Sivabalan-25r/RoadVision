# RoadVision Backend (Python)

AI-powered license plate detection and validation backend.

## Pipeline

```
Video Upload → Frame Extraction → YOLO Plate Detection → Geometric Filtering
→ Plate Crop Preprocessing → CRNN Text Recognition → Indian Plate Validation
→ JSON Response
```

## Required Files

| File | Required | Description |
|------|----------|-------------|
| `models/license_plate_detector.pt` | **YES** | YOLO license plate detector weights. Server will not start without this. |
| `models/crnn.pth` | Recommended | CRNN text recognition weights. Without this, plate text cannot be read. |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place model files
#    Copy license_plate_detector.pt → backend-python/models/
#    Copy crnn.pth → backend-python/models/

# 3. Start the server
cd backend-python
uvicorn main:app --reload --port 8000
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze-video` | POST | Upload video for plate detection |
| `/docs` | GET | Interactive API documentation |

### POST /analyze-video

Upload a video file (MP4, MOV, WEBM). Returns:

```json
{
  "detections": [
    {
      "detected_plate": "KA01AB1234",
      "correct_plate": "KA 01 AB 1234",
      "violation": null,
      "confidence": 0.91,
      "frame": 120,
      "bbox": [312, 210, 140, 48]
    }
  ]
}
```

## Configuration

| Setting | Value | Location |
|---------|-------|----------|
| YOLO confidence | 0.5 | `recognition/plate_reader.py` |
| Min aspect ratio | 2.0 | `recognition/plate_reader.py` |
| Max aspect ratio | 6.0 | `recognition/plate_reader.py` |
| Min plate size | 80×25 px | `recognition/plate_reader.py` |
| Max width ratio | 50% of frame | `recognition/plate_reader.py` |
| Min Y position | 45% of frame | `recognition/plate_reader.py` |
