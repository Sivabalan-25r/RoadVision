# RoadVision Tech Stack Upgrade — Design Document

## Overview

This design document specifies the complete technical architecture for upgrading RoadVision's license plate recognition pipeline from YOLOv8n + CRNN to a modern, high-performance stack: **YOLOv26 + BoT-SORT + PaddleOCR (PP-OCRv5)** with grammar validation and Levenshtein distance deduplication.

The upgrade maintains full backward compatibility with the existing FastAPI backend and HTML/CSS/JS frontend while delivering:
- **Higher detection accuracy**: YOLOv26 with improved training data and architecture
- **Reliable multi-object tracking**: BoT-SORT prevents duplicate detections across frames
- **Superior OCR performance**: PaddleOCR PP-OCRv5 with ensemble preprocessing variants
- **Intelligent validation**: Grammar rules + character correction + format validation
- **Deduplication**: Levenshtein distance-based fuzzy matching to eliminate near-duplicates

---

## Architecture

### High-Level Pipeline Flow

```
Video/Frame Input
    ↓
[Frame Extraction] (every 3rd frame for videos)
    ↓
[YOLOv26 Detection] (confidence ≥ 0.25, 320×320 inference)
    ↓
[Geometric Filtering] (aspect ratio, area, size constraints)
    ↓
[Bounding Box Refinement] (remove vehicle body padding)
    ↓
[Plate Preprocessing] (bilateral filter → CLAHE → sharpen → threshold)
    ↓
[BoT-SORT Tracking] (assign track IDs, maintain consistency)
    ↓
[PaddleOCR Ensemble] (4 variants: original, CLAHE, sharpened, thresholded)
    ↓
[Text Cleaning] (normalize, correct common OCR errors)
    ↓
[Position-Based Corrections] (state code letters, district digits, etc.)
    ↓
[Grammar Validation] (Indian RTO format: AA NN AA NNNN)
    ↓
[Character Manipulation Detection] (flag suspicious substitutions)
    ↓
[Font Anomaly Detection] (flag non-standard fonts)
    ↓
[Vehicle Registration Lookup] (SQLite query)
    ↓
[Confidence Scoring] (weighted YOLO + OCR + format boost + stability boost)
    ↓
[Plate Stabilization] (require ≥2 frames for confirmation)
    ↓
[Levenshtein Deduplication] (distance ≤2 → merge, keep highest confidence)
    ↓
[Database Persistence] (SQLite insert)
    ↓
[API Response] (JSON with detection results, base64 plate images)
```

### System Components

#### 1. Detection Module (YOLOv26)
- **Model**: `license_plate_detector.pt` (YOLOv26 trained on license plates)
- **Input**: Video frames (any resolution)
- **Output**: Bounding boxes with confidence scores
- **Configuration**:
  - Confidence threshold: 0.25 (low to catch more plates, false positives filtered downstream)
  - Image size: 320×320 (balance between speed and accuracy)
  - FP16 half-precision: Auto-enabled on CUDA GPU
  - Inference time: <100ms per frame on standard CPU

#### 2. Tracking Module (BoT-SORT)
- **Algorithm**: Bag of Tricks for SORT (advanced multi-object tracking)
- **Input**: YOLOv26 detections (bounding boxes + confidence)
- **Output**: Track IDs assigned to each detection
- **Configuration**:
  - Max age: 30 frames (track expires after 30 frames without detection)
  - Min hits: 1 (assign track ID immediately)
  - IOU threshold: 0.1 (intersection-over-union for matching)
- **Purpose**: Maintain consistent track IDs across frames to enable plate stabilization

#### 3. Preprocessing Module
- **Input**: Plate crop (raw BGR image)
- **Output**: 4 preprocessed variants for ensemble OCR
- **Techniques**:
  - Bilateral filtering (noise reduction while preserving edges)
  - CLAHE (Contrast Limited Adaptive Histogram Equalization for shadows/glare)
  - Sharpening kernel (enhance character edges)
  - Adaptive thresholding (Gaussian, Mean, OTSU methods)
- **Target size**: 320×120 pixels (upscaled with cubic interpolation if smaller)

#### 4. OCR Module (PaddleOCR PP-OCRv5)
- **Primary**: PaddleOCR PP-OCRv5 (Baidu's latest OCR engine)
- **Fallback chain**: CRNN → EasyOCR → Tesseract
- **Input**: Preprocessed plate image
- **Output**: Recognized text with confidence score
- **Confidence threshold**: 0.25 (minimum confidence to accept OCR result)
- **Ensemble strategy**: Run OCR on 4 variants, select highest confidence result

#### 5. Validation Module
- **Grammar Validation**: Indian RTO format (AA NN AA NNNN)
- **Character Correction**: Position-based substitution (0→O, 1→I, etc.)
- **Character Manipulation Detection**: Flag suspicious substitutions
- **Font Anomaly Detection**: Detect non-standard fonts
- **Vehicle Registration Lookup**: Query SQLite for registered vehicles
- **Violation Detection**: Unregistered, character manipulation, font anomaly

#### 6. Confidence Scoring Module
- **Formula**: `(YOLO_conf × 0.4 + OCR_conf × 0.6) × format_boost × stability_boost`
- **Format boost**: 1.15× if plate matches Indian RTO format
- **Stability boost**: +0.05 per additional frame (max 1.0)
- **Final cap**: 1.0 (normalized to [0.0, 1.0])
- **Thresholds**:
  - Low confidence: < 0.50
  - High confidence: ≥ 0.50

#### 7. Stabilization Module
- **Purpose**: Reduce false positives by requiring multi-frame confirmation
- **Requirement**: Plate must be detected in ≥2 consecutive frames
- **Expiry**: Tracker entries expire after 30 seconds without detection
- **Output**: "Detecting..." status until confirmed, then full detection data

#### 8. Deduplication Module
- **Algorithm**: Levenshtein distance-based fuzzy matching
- **Threshold**: Distance ≤2 AND length difference ≤2 → duplicate
- **Action**: Keep detection with highest combined confidence
- **Logging**: Log original and duplicate plate with distance metric

#### 9. Database Module (SQLite)
- **Tables**: `detections`, `cameras`, `vehicle_registrations`
- **Persistence**: All confirmed detections saved immediately
- **Queries**: Filter by camera ID, violation status, timestamp range
- **Indexes**: camera_id, timestamp, violation for fast queries

#### 10. API Module (FastAPI)
- **Endpoints**:
  - `POST /analyze-video`: Upload video, return detections
  - `POST /api/process-frame`: Process single camera frame
  - `GET /api/live-detections`: Get live detections (mock for demo)
  - `GET /api/cameras`: List all cameras
  - `GET /api/cameras/{camera_id}`: Get camera info
  - `PUT /api/cameras/{camera_id}`: Update camera info
  - `GET /api/detections/{camera_id}`: Get detections for camera
  - `POST /api/detections/{camera_id}`: Add detection
  - `DELETE /api/detections/{camera_id}`: Clear detections
- **Response format**: JSON with backward-compatible structure

---

## Components and Interfaces

### Detection Component

**Interface**: `detect_plates(frame: np.ndarray, frame_number: int) → List[Dict]`

**Returns**:
```python
[
  {
    'bbox': [x, y, w, h],           # Bounding box in pixels
    'crop': np.ndarray,              # Preprocessed 320×120 image
    'raw_crop': np.ndarray,          # Original BGR crop
    'confidence': float              # YOLO confidence (0.0-1.0)
  },
  ...
]
```

### Tracking Component

**Interface**: `BoT-SORT(detections: List[Dict]) → List[Dict]`

**Returns**: Same detections with added `track_id` field

### OCR Component

**Interface**: `read_plate(plate_image: np.ndarray, preprocessed_image: np.ndarray) → Optional[str]`

**Returns**: Recognized plate text (e.g., "WB65D18753") or None

### Validation Component

**Interface**: `validate_plate(text: str, image: np.ndarray) → ValidationResult`

**Returns**:
```python
ValidationResult(
  detected_plate: str,              # Raw OCR output
  correct_plate: Optional[str],     # Corrected text (if violations found)
  violation: Optional[str],         # Violation type or None
  font_anomaly: Optional[bool],     # Font anomaly flag
  confidence_modifier: float,       # Confidence multiplier (0.0-1.0)
  vehicle_info: Optional[Dict]      # Registration info or None
)
```

### Confidence Scoring Component

**Interface**: `calculate_confidence(yolo_conf: float, ocr_conf: float, validation: ValidationResult, frames_seen: int) → float`

**Returns**: Combined confidence score (0.0-1.0)

### Stabilization Component

**Interface**: `stabilize_detection(normalized_plate: str, detection: Dict, camera_id: str) → bool`

**Returns**: True if plate is confirmed (≥2 frames), False otherwise

### Deduplication Component

**Interface**: `deduplicate_detections(detections: List[Dict]) → List[Dict]`

**Returns**: Deduplicated detections (Levenshtein distance ≤2 merged)

---

## Data Models

### Detection Record (Database)

```sql
CREATE TABLE detections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  camera_id TEXT NOT NULL,
  detected_plate TEXT NOT NULL,
  correct_plate TEXT,
  violation TEXT,
  font_anomaly BOOLEAN,
  confidence REAL NOT NULL,
  frame_number INTEGER,
  bbox TEXT,                        -- JSON: [x, y, w, h]
  plate_image BLOB,                 -- JPEG binary
  vehicle_info TEXT,                -- JSON: {owner, registration_date, status}
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  source TEXT DEFAULT 'video_analysis'
);
```

### Camera Record (Database)

```sql
CREATE TABLE cameras (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  location TEXT,
  resolution TEXT DEFAULT '1080p',
  latitude TEXT,
  longitude TEXT,
  address TEXT,
  accuracy TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Vehicle Registration Record (Database)

```sql
CREATE TABLE vehicle_registrations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  plate TEXT UNIQUE NOT NULL,
  owner TEXT,
  registration_date TEXT,
  status TEXT DEFAULT 'active',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### API Response Format (Backward Compatible)

```json
{
  "detections": [
    {
      "detected_plate": "WB65D18753",
      "correct_plate": "WB 65 D 18753",
      "violation": null,
      "font_anomaly": false,
      "confidence": 0.92,
      "frame": 0,
      "bbox": [35, 45, 18, 8],
      "plate_image": "data:image/jpeg;base64,...",
      "frames_seen": 2,
      "vehicle_info": {
        "owner": "John Doe",
        "registration_date": "2023-01-15",
        "status": "active"
      }
    }
  ]
}
```

---

## Error Handling and Fallback Strategies

### Detection Failures

| Scenario | Handling |
|----------|----------|
| Model file missing | Raise `FileNotFoundError` at startup, prevent server start |
| CUDA unavailable | Fall back to CPU inference (slower but functional) |
| Frame decode error | Log warning, skip frame, continue processing |
| YOLO inference error | Log error, return empty detection list |

### OCR Failures

| Scenario | Handling |
|----------|----------|
| PaddleOCR unavailable | Fall back to CRNN (if weights available) |
| CRNN unavailable | Fall back to EasyOCR |
| EasyOCR unavailable | Fall back to Tesseract |
| All OCR engines unavailable | Log error, return empty string, skip detection |
| OCR confidence < 0.25 | Reject result, try next variant |

### Tracking Failures

| Scenario | Handling |
|----------|----------|
| BoT-SORT initialization error | Log error, continue without tracking (frame-level aggregation only) |
| Track ID assignment error | Assign sequential IDs as fallback |

### Validation Failures

| Scenario | Handling |
|----------|----------|
| Invalid plate format | Mark as violation, return corrected text if possible |
| Character manipulation detected | Flag violation, return both original and corrected |
| Font anomaly detected | Flag violation, continue processing |
| Registration lookup error | Log warning, continue without vehicle info |

### Database Failures

| Scenario | Handling |
|----------|----------|
| Database connection error | Log error, continue processing (in-memory only) |
| Insert error | Log warning, skip persistence, continue |
| Query error | Log error, return empty results |

---

## Performance Considerations and Optimization Points

### Detection Performance

- **YOLOv26 inference**: ~50-100ms per frame on CPU, ~10-20ms on GPU
- **Optimization**: Use FP16 half-precision on CUDA GPU (2-3× speedup)
- **Batching**: Process multiple frames in parallel (future enhancement)
- **Frame sampling**: Process every 3rd frame for videos (3× speedup)

### OCR Performance

- **PaddleOCR inference**: ~50-100ms per plate on CPU
- **Optimization**: Ensemble on 4 variants (parallel processing possible)
- **Caching**: Cache OCR model in memory (loaded once at startup)
- **Preprocessing**: Bilateral filter + CLAHE faster than NLM denoising

### Tracking Performance

- **BoT-SORT**: ~5-10ms per frame (minimal overhead)
- **Optimization**: Efficient distance matrix computation
- **Memory**: Track history stored in memory (expires after 30s)

### Database Performance

- **Indexes**: Create indexes on `camera_id`, `timestamp`, `violation`
- **Batch inserts**: Group multiple detections for faster insertion
- **Query optimization**: Use indexed columns in WHERE clauses

### Memory Usage

- **Model loading**: ~500MB for YOLOv26 + PaddleOCR + CRNN
- **Frame buffering**: Store only current frame (not entire video)
- **Track history**: Expire entries after 30 seconds
- **Database**: SQLite in-process (no network overhead)

### Throughput

- **Video processing**: ~10-15 FPS (with frame sampling)
- **Live monitoring**: ~5-10 FPS per camera (real-time)
- **Concurrent cameras**: 2-3 cameras on standard CPU

---

## API Response Format (Maintaining Backward Compatibility)

### Video Analysis Endpoint

**Request**: `POST /analyze-video`
```
Content-Type: multipart/form-data
Body: video file (MP4, MOV, WEBM)
```

**Response**: 200 OK
```json
{
  "detections": [
    {
      "detected_plate": "WB65D18753",
      "correct_plate": "WB 65 D 18753",
      "violation": null,
      "font_anomaly": false,
      "confidence": 0.92,
      "frame": 0,
      "bbox": [35, 45, 18, 8],
      "plate_image": "data:image/jpeg;base64,...",
      "frames_seen": 2,
      "vehicle_info": {...}
    }
  ]
}
```

### Live Frame Processing Endpoint

**Request**: `POST /api/process-frame`
```
Content-Type: multipart/form-data
Body: image file (JPEG, PNG)
       original_width: 1920
       original_height: 1080
       camera_id: CAM-001
```

**Response**: 200 OK
```json
{
  "detections": [
    {
      "detected_plate": "Detecting...",
      "correct_plate": null,
      "violation": null,
      "confidence": 0.85,
      "bbox": [35.5, 45.2, 18.3, 8.1],
      "plate_image": "data:image/jpeg;base64,..."
    }
  ]
}
```

### Detection History Endpoint

**Request**: `GET /api/detections/{camera_id}?violations_only=false&limit=100`

**Response**: 200 OK
```json
{
  "detections": [
    {
      "detected_plate": "WB65D18753",
      "correct_plate": "WB 65 D 18753",
      "violation": null,
      "confidence": 0.92,
      "timestamp": "2024-01-15T10:30:45Z"
    }
  ]
}
```

---

## Configuration Management

### Model Configuration

**File**: `backend-python/config.py` (to be created)

```python
# Detection
YOLO_MODEL_PATH = "backend-python/models/license_plate_detector.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IMAGE_SIZE = 320
YOLO_USE_HALF_PRECISION = True  # Auto-detect CUDA

# Tracking
BOTSORT_MAX_AGE = 30
BOTSORT_MIN_HITS = 1
BOTSORT_IOU_THRESHOLD = 0.1

# OCR
OCR_CONFIDENCE_THRESHOLD = 0.25
OCR_FALLBACK_CHAIN = ["paddleocr", "crnn", "easyocr", "tesseract"]

# Validation
PLATE_FORMAT_PATTERN = r"^[A-Z]{2}\d{2}[A-Z0-9]{2,8}$"
MIN_PLATE_LENGTH = 6
MAX_PLATE_LENGTH = 12

# Stabilization
STABILIZATION_FRAMES = 2
TRACKER_EXPIRY_SECONDS = 30

# Deduplication
LEVENSHTEIN_THRESHOLD = 2
LENGTH_DIFF_THRESHOLD = 2

# Confidence Scoring
YOLO_WEIGHT = 0.4
OCR_WEIGHT = 0.6
FORMAT_BOOST = 1.15
STABILITY_BOOST_PER_FRAME = 0.05

# Video Processing
FRAME_INTERVAL = 3  # Process every 3rd frame
```

### Threshold Configuration

All thresholds are configurable via environment variables or config file:

```bash
YOLO_CONF=0.25
OCR_CONF=0.25
STABILIZATION_FRAMES=2
LEVENSHTEIN_THRESHOLD=2
```

---

## Logging and Monitoring Strategy

### Logging Levels

- **INFO**: Model loading, video processing start/end, detection counts
- **DEBUG**: Frame-level details, OCR variants, confidence calculations
- **WARNING**: Fallback usage, low confidence detections, DB errors
- **ERROR**: Model loading failures, OCR pipeline errors, critical failures

### Log Format

```
2024-01-15 10:30:45 [INFO] roadvision: Loaded YOLOv26 model (parameters: 68.2M)
2024-01-15 10:30:46 [INFO] roadvision: Processing video: traffic_cam_001.mp4
2024-01-15 10:30:47 [DEBUG] roadvision: Frame 0: Detected 3 valid plates
2024-01-15 10:30:48 [INFO] roadvision: Plate 'WB65D18753': seen in 2 frames, confidence 0.92
2024-01-15 10:30:49 [INFO] roadvision: Final results: 2 unique detections, 0 violations
```

### Monitoring Metrics

- **Detection rate**: Plates detected per frame
- **OCR success rate**: Successful OCR reads / total detections
- **Stabilization rate**: Confirmed plates / total detections
- **Deduplication rate**: Duplicates removed / total detections
- **Inference time**: Average time per frame
- **Database latency**: Average insert time
- **API response time**: Average response time per endpoint

### Health Checks

- **Model availability**: Check if YOLOv26 model is loaded
- **OCR availability**: Check if at least one OCR engine is available
- **Database connectivity**: Check if SQLite is accessible
- **Disk space**: Check if sufficient space for logs and database



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: YOLOv26 Detection Returns Required Fields

*For any* video frame, when YOLOv26 detects a license plate, the detection result SHALL include both bounding box coordinates and a confidence score in the range [0.0, 1.0].

**Validates: Requirements 1.2, 1.3**

### Property 2: BoT-SORT Assigns Unique Track IDs

*For any* set of detections in a single frame, BoT-SORT SHALL assign a unique track ID to each detection, with no two detections sharing the same ID.

**Validates: Requirements 2.3, 2.6**

### Property 3: Track ID Consistency Across Frames

*For any* plate detected in two consecutive frames with sufficient overlap, BoT-SORT SHALL assign the same track ID to both detections.

**Validates: Requirements 2.4**

### Property 4: Track Expiry After Inactivity

*For any* track that has not received a detection for 30 consecutive frames, the tracking system SHALL mark the track as inactive and stop assigning its ID to new detections.

**Validates: Requirements 2.5**

### Property 5: OCR Returns Text and Confidence

*For any* plate crop image, PaddleOCR SHALL return recognized text along with a confidence score in the range [0.0, 1.0].

**Validates: Requirements 3.2**

### Property 6: OCR Confidence Thresholding

*For any* OCR result with confidence below 0.25, the system SHALL reject the result and return an empty string.

**Validates: Requirements 3.3, 3.4**

### Property 7: Ensemble OCR Selects Highest Confidence

*For any* set of OCR variants processed on the same plate crop, the ensemble strategy SHALL select the variant with the highest confidence score.

**Validates: Requirements 3.5**

### Property 8: Grammar Validation Against Indian Format

*For any* text string, the grammar validator SHALL determine whether it matches the Indian RTO license plate format (AA NN AA NNNN) and return a validation status.

**Validates: Requirements 4.1**

### Property 9: Position-Based Character Corrections

*For any* plate text with character errors in specific positions, the validator SHALL apply position-based corrections: state code (positions 0-1) corrected to letters, district code (positions 2-3) corrected to digits, and series/registration corrected accordingly.

**Validates: Requirements 4.4, 4.5, 4.6, 4.7**

### Property 10: Plate Length Validation

*For any* text string with length < 6 or > 12 characters, the validator SHALL reject it as invalid.

**Validates: Requirements 4.8**

### Property 11: Levenshtein Distance Symmetry

*For any* two plate strings A and B, the Levenshtein distance between A and B SHALL equal the Levenshtein distance between B and A (symmetry property).

**Validates: Requirements 5.1**

### Property 12: Duplicate Detection by Levenshtein Distance

*For any* two plates with Levenshtein distance ≤ 2 AND length difference ≤ 2, the deduplicator SHALL mark them as duplicates.

**Validates: Requirements 5.2**

### Property 13: Deduplication Selects Highest Confidence

*For any* set of duplicate detections, the deduplicator SHALL keep only the detection with the highest combined confidence score.

**Validates: Requirements 5.3, 5.6**

### Property 14: Distinct Plates Not Merged

*For any* two plates with Levenshtein distance > 2, the deduplicator SHALL treat them as distinct and not merge them.

**Validates: Requirements 5.5**

### Property 15: API Response Contains Required Fields

*For any* detection returned by the backend API, the response SHALL include all required fields: `detected_plate`, `correct_plate`, `violation`, `confidence`, `bbox`, and `plate_image`.

**Validates: Requirements 6.5**

### Property 16: Bounding Boxes as Percentages

*For any* bounding box returned by the frame processor, all coordinate values SHALL be in the range [0.0, 100.0] representing percentages of the frame dimensions.

**Validates: Requirements 6.7, 11.2**

### Property 17: Plate Crop Upscaling

*For any* plate crop smaller than 320×120 pixels, the preprocessor SHALL upscale it to at least 320×120 using cubic interpolation.

**Validates: Requirements 7.5**

### Property 18: Preprocessing Output Dimensions

*For any* plate crop processed by the preprocessor, the output SHALL be a binary image with dimensions 320×120 pixels.

**Validates: Requirements 7.6**

### Property 19: Preprocessing Generates Four Variants

*For any* plate crop, the preprocessing module SHALL generate exactly 4 variants: original, CLAHE-only, sharpened, and thresholded.

**Validates: Requirements 7.7**

### Property 20: Plate Stabilization Requires Multiple Frames

*For any* plate detected in only 1 frame, the stabilizer SHALL NOT save it to the database until it is confirmed in ≥2 frames.

**Validates: Requirements 8.2, 8.5**

### Property 21: Stabilization Expiry After Timeout

*For any* plate tracker entry that has not received a detection for 30 seconds, the stabilizer SHALL expire the entry and stop tracking it.

**Validates: Requirements 8.4**

### Property 22: Confirmed Plates Include frames_seen Field

*For any* confirmed plate returned to the frontend, the response SHALL include a `frames_seen` field indicating the number of frames in which the plate was detected.

**Validates: Requirements 8.6**

### Property 23: Unconfirmed Plates Show "Detecting..."

*For any* plate detected in only 1 frame, the frame processor SHALL return "Detecting..." as the `detected_plate` value until the plate is confirmed in ≥2 frames.

**Validates: Requirements 8.7, 11.5**

### Property 24: Confidence Score Calculation

*For any* detection, the combined confidence SHALL be calculated as: `(YOLO_conf × 0.4 + OCR_conf × 0.6) × format_boost × stability_boost`, capped at 1.0.

**Validates: Requirements 9.1, 9.4**

### Property 25: Format Boost Applied to Valid Plates

*For any* plate matching the Indian RTO format, the confidence scorer SHALL apply a format boost of 1.15× to the base confidence.

**Validates: Requirements 9.2**

### Property 26: Stability Boost for Multi-Frame Detections

*For any* plate confirmed in N frames (N ≥ 2), the confidence scorer SHALL apply a stability boost of +0.05 per additional frame, capped at 1.0.

**Validates: Requirements 9.3**

### Property 27: Confidence Rounding to Two Decimals

*For any* calculated confidence score, the final value SHALL be rounded to exactly 2 decimal places.

**Validates: Requirements 9.7**

### Property 28: Video Frame Extraction at Interval

*For any* video processed with frame interval N, the video processor SHALL extract and process every Nth frame.

**Validates: Requirements 10.1**

### Property 29: Frame Numbers Recorded with Detections

*For any* detection returned from video processing, the response SHALL include the frame number in which the detection occurred.

**Validates: Requirements 10.3, 10.4**

### Property 30: Invalid Video Returns Empty Detections

*For any* invalid or corrupted video file, the video processor SHALL return an empty detection list.

**Validates: Requirements 10.5**

### Property 31: Video Processing Continues on Frame Errors

*For any* video with one or more unreadable frames, the video processor SHALL continue processing remaining frames and return detections from readable frames.

**Validates: Requirements 10.6**

### Property 32: Frame Processor Returns Percentages

*For any* frame processed by the frame processor, all bounding box coordinates SHALL be returned as percentages (0-100) of the frame dimensions.

**Validates: Requirements 11.2**

### Property 33: OCR Integration in Frame Processing

*For any* plate detected in a frame, the frame processor SHALL run OCR and return the recognized text in the response.

**Validates: Requirements 11.3**

### Property 34: Validation Integration in Frame Processing

*For any* OCR result, the frame processor SHALL validate the plate and include the violation status in the response.

**Validates: Requirements 11.4**

### Property 35: Bounding Box Scaling to Original Resolution

*For any* frame with provided original dimensions, the frame processor SHALL scale bounding boxes from processing resolution to original resolution such that they remain within frame bounds [0, 100].

**Validates: Requirements 11.7**

### Property 36: Database Insertion of Confirmed Detections

*For any* confirmed detection, the database module SHALL insert a record with all required fields: plate text, confidence, frame number, camera ID, violation status, and timestamp.

**Validates: Requirements 12.1, 12.2**

### Property 37: Database Query Sorting by Timestamp

*For any* query of detections from the database, results SHALL be sorted by timestamp in descending order (newest first).

**Validates: Requirements 12.3**

### Property 38: Database Filtering by Camera and Violation

*For any* query with camera ID and violation status filters, the database SHALL return only detections matching both criteria.

**Validates: Requirements 12.6**

### Property 39: Plate Parsing Extracts Components

*For any* valid Indian license plate string, the parser SHALL extract state code (2 letters), district code (2 digits), series (1-3 letters), and registration number (1-4 digits) into separate fields.

**Validates: Requirements 15.1, 15.2**

### Property 40: Plate Pretty-Printing Format

*For any* parsed plate object, the pretty-printer SHALL format it as "AA NN AA NNNN" with spaces between components, matching the Indian RTO format.

**Validates: Requirements 15.3, 15.7**

### Property 41: Plate Parsing Round-Trip

*For any* valid Indian license plate, parsing then pretty-printing then parsing again SHALL produce an equivalent object (parse → print → parse identity).

**Validates: Requirements 15.4**

### Property 42: Character Manipulation Detection

*For any* plate text containing common character substitutions (0→O, 1→I, 8→B, etc.), the validator SHALL detect the manipulation and flag it.

**Validates: Requirements 16.1, 16.2**

### Property 43: Character Manipulation Correction

*For any* plate with detected character manipulation, the validator SHALL apply corrections and return both the original and corrected text.

**Validates: Requirements 16.3, 16.4**

### Property 44: Font Anomaly Detection Output

*For any* plate image analyzed for font anomalies, the analyzer SHALL return a boolean flag: true if anomalies detected, false if standard font, null if analysis fails.

**Validates: Requirements 17.3, 17.4, 17.5**

### Property 45: Vehicle Registration Lookup

*For any* detected plate, the registration lookup module SHALL query the database and return vehicle info if found, or null if not found.

**Validates: Requirements 18.1, 18.2, 18.3**

### Property 46: Unregistered Vehicle Violation

*For any* plate not found in the vehicle registration database, the validator SHALL flag it with "Unregistered_Vehicle" violation.

**Validates: Requirements 18.4**

### Property 47: Registered Vehicle Info Included

*For any* plate found in the vehicle registration database, the detection result SHALL include vehicle info with owner, registration date, and status.

**Validates: Requirements 18.5, 18.6**

### Property 48: RTO Format Validation

*For any* plate text, the RTO validator SHALL check it against Indian license plate format rules and return appropriate violation type or null.

**Validates: Requirements 19.1, 19.2, 19.3**

### Property 49: RTO Violation Types

*For any* invalid plate, the RTO validator SHALL return one of the following violation types: "Character_Manipulation", "Font_Anomaly", "Unregistered_Vehicle", or "Invalid_Format".

**Validates: Requirements 19.4, 19.5, 19.6**

### Property 50: Valid Plate Returns No Violation

*For any* plate matching Indian RTO format and registered in the database, the validator SHALL return `violation: null` and `correct_plate: null`.

**Validates: Requirements 19.7**

### Property 51: Image Encoding to Data URL

*For any* plate crop image, the encoder SHALL encode it as JPEG and return a data URL in the format `data:image/jpeg;base64,...`.

**Validates: Requirements 20.2**

### Property 52: Image Encoding Error Handling

*For any* plate crop that fails to encode, the encoder SHALL return null and log a warning.

**Validates: Requirements 20.3**

### Property 53: Detection Response Includes Plate Image

*For any* detection returned to the frontend, the response SHALL include a `plate_image` field with the base64-encoded plate crop.

**Validates: Requirements 20.5**



## Testing Strategy

### Dual Testing Approach

This feature requires both **unit tests** and **property-based tests** for comprehensive coverage:

- **Unit Tests**: Verify specific examples, edge cases, and error conditions
  - Model loading and initialization
  - Fallback OCR engine selection
  - Database operations (insert, query, update)
  - API endpoint responses
  - Error handling for invalid inputs

- **Property-Based Tests**: Verify universal properties across all inputs
  - Detection field presence and format
  - Confidence score calculation and bounds
  - Levenshtein distance symmetry
  - Plate parsing round-trip
  - Bounding box scaling and percentages
  - Stabilization and deduplication logic

### Property-Based Testing Configuration

**Testing Library**: Hypothesis (Python)

**Test Configuration**:
- Minimum 100 iterations per property test
- Each test references its corresponding design property
- Tag format: `Feature: tech-stack-upgrade, Property {number}: {property_text}`

### Unit Test Coverage

**Model Loading Tests**:
- Test YOLOv26 model loads successfully from correct path
- Test FileNotFoundError raised when model missing
- Test CUDA detection and FP16 enablement
- Test OCR fallback chain (PaddleOCR → CRNN → EasyOCR → Tesseract)

**Detection Tests**:
- Test YOLO detection returns required fields (bbox, confidence)
- Test geometric filtering rejects invalid plates
- Test bounding box refinement removes vehicle body padding
- Test detection confidence in range [0.0, 1.0]

**Tracking Tests**:
- Test BoT-SORT initialization
- Test unique track ID assignment
- Test track ID consistency across frames
- Test track expiry after 30 frames

**OCR Tests**:
- Test PaddleOCR returns text and confidence
- Test confidence thresholding (< 0.25 rejected)
- Test ensemble variant selection
- Test fallback to CRNN/EasyOCR/Tesseract

**Validation Tests**:
- Test Indian RTO format validation
- Test position-based character corrections
- Test character manipulation detection
- Test font anomaly detection
- Test vehicle registration lookup

**Confidence Scoring Tests**:
- Test base confidence calculation (YOLO 0.4 + OCR 0.6)
- Test format boost (1.15× for valid plates)
- Test stability boost (+0.05 per frame)
- Test confidence capping at 1.0
- Test rounding to 2 decimal places

**Stabilization Tests**:
- Test single-frame detections not saved
- Test multi-frame confirmation (≥2 frames)
- Test tracker expiry after 30 seconds
- Test "Detecting..." status for unconfirmed plates

**Deduplication Tests**:
- Test Levenshtein distance calculation
- Test duplicate detection (distance ≤ 2)
- Test highest confidence selection
- Test distinct plates not merged

**Database Tests**:
- Test detection insertion with all fields
- Test camera creation and update
- Test query filtering by camera and violation
- Test timestamp sorting

**API Tests**:
- Test `/analyze-video` endpoint response format
- Test `/api/process-frame` endpoint response format
- Test `/api/detections/{camera_id}` endpoint
- Test bounding boxes as percentages
- Test base64 plate image encoding

**Error Handling Tests**:
- Test invalid video file handling
- Test corrupted frame handling
- Test missing model file handling
- Test database connection errors
- Test OCR engine unavailability

### Property-Based Test Examples

**Property 1: YOLOv26 Detection Returns Required Fields**
```python
@given(frame=generate_video_frame())
def test_yolo_detection_fields(frame):
    detections = detect_plates(frame)
    for detection in detections:
        assert 'bbox' in detection
        assert 'confidence' in detection
        assert 0.0 <= detection['confidence'] <= 1.0
```

**Property 11: Levenshtein Distance Symmetry**
```python
@given(plate_a=generate_plate_string(), plate_b=generate_plate_string())
def test_levenshtein_symmetry(plate_a, plate_b):
    dist_ab = levenshtein_distance(plate_a, plate_b)
    dist_ba = levenshtein_distance(plate_b, plate_a)
    assert dist_ab == dist_ba
```

**Property 24: Confidence Score Calculation**
```python
@given(
    yolo_conf=st.floats(0.0, 1.0),
    ocr_conf=st.floats(0.0, 1.0),
    frames_seen=st.integers(1, 10)
)
def test_confidence_calculation(yolo_conf, ocr_conf, frames_seen):
    result = calculate_confidence(yolo_conf, ocr_conf, frames_seen)
    assert 0.0 <= result <= 1.0
    assert result == round(result, 2)
```

**Property 41: Plate Parsing Round-Trip**
```python
@given(plate=generate_valid_indian_plate())
def test_plate_parsing_roundtrip(plate):
    parsed = parse_plate(plate)
    pretty = pretty_print_plate(parsed)
    reparsed = parse_plate(pretty)
    assert parsed == reparsed
```

**Property 51: Image Encoding to Data URL**
```python
@given(plate_crop=generate_plate_crop())
def test_image_encoding_format(plate_crop):
    encoded = encode_plate_image(plate_crop)
    assert encoded.startswith('data:image/jpeg;base64,')
    assert len(encoded) > 30
```

### Test Execution

**Unit Tests**:
```bash
pytest backend-python/tests/test_detection.py -v
pytest backend-python/tests/test_tracking.py -v
pytest backend-python/tests/test_ocr.py -v
pytest backend-python/tests/test_validation.py -v
pytest backend-python/tests/test_database.py -v
pytest backend-python/tests/test_api.py -v
```

**Property-Based Tests**:
```bash
pytest backend-python/tests/test_properties.py -v --hypothesis-seed=0
```

### Coverage Goals

- **Detection module**: 90%+ coverage
- **Tracking module**: 85%+ coverage
- **OCR module**: 80%+ coverage (due to external dependencies)
- **Validation module**: 95%+ coverage
- **Database module**: 90%+ coverage
- **API module**: 85%+ coverage
- **Overall**: 85%+ coverage

