# RoadVision Tech Stack Upgrade — Requirements Document

## Introduction

This document specifies the requirements for upgrading the RoadVision license plate recognition system to a modern, high-performance tech stack. The system will completely replace the current detection, tracking, and OCR pipeline with:

- **Detection**: YOLOv26 (replaces YOLOv8n)
- **Tracking**: BoT-SORT (replaces basic frame-to-frame tracking)
- **OCR**: PaddleOCR (PP-OCRv5) (replaces CRNN)
- **Post-processing**: Grammar validation + Levenshtein distance correction (new)

The upgrade maintains full backward compatibility with the existing HTML/CSS/JS frontend while significantly improving detection accuracy, tracking reliability, and OCR performance. The system continues to serve the FastAPI backend with SQLite persistence for registration and detection history.

---

## Glossary

- **YOLOv26**: Latest YOLO object detection model family (new)
- **YOLOv8n**: Nano variant of YOLOv8 (old model, being removed)
- **BoT-SORT**: Bag of Tricks for SORT — advanced multi-object tracking algorithm
- **SORT**: Simple Online and Realtime Tracking (baseline tracking algorithm)
- **PaddleOCR**: Baidu's PP-OCRv5 optical character recognition engine
- **PP-OCRv5**: Latest version of PaddleOCR with improved accuracy
- **CRNN**: Convolutional Recurrent Neural Network (current OCR recognizer)
- **Levenshtein_Distance**: Edit distance metric for string similarity (insertions, deletions, substitutions)
- **Grammar_Validation**: Rule-based validation of Indian license plate format (AA NN AA NNNN)
- **Detection_Confidence**: YOLO model confidence score (0.0–1.0)
- **OCR_Confidence**: PaddleOCR confidence score (0.0–1.0)
- **Combined_Confidence**: Weighted average of detection and OCR confidence
- **Plate_Stabilization**: Multi-frame validation requiring consistent plate readings across ≥2 frames
- **Backward_Compatibility**: Frontend continues to work without modification
- **RTO_Rules**: Indian Road Transport Office vehicle registration and plate format regulations
- **License_Plate_Detector**: Dedicated YOLOv26 model trained on license plate detection
- **Video_Pipeline**: End-to-end processing: frame extraction → detection → tracking → OCR → validation
- **Live_Monitoring**: Real-time camera feed processing with frame-by-frame analysis
- **Detection_History**: SQLite database records of all detected plates with metadata
- **Violation_Detection**: Identification of illegal plate formats (character manipulation, font anomalies)

---

## Requirements

### Requirement 1: Upgrade YOLOv26 Detection Model

**User Story:** As a system operator, I want the license plate detection to use the latest YOLOv26 model instead of YOLOv8n, so that I can achieve higher detection accuracy and speed on modern hardware.

#### Acceptance Criteria

1. WHEN the backend starts, THE Detection_System SHALL load the YOLOv26 model from `backend-python/models/license_plate_detector.pt`
2. WHEN a video frame is processed, THE Detection_System SHALL run YOLOv26 inference with confidence threshold ≥ 0.25 and image size 320×320
3. WHEN YOLOv26 detects a license plate, THE Detection_System SHALL return bounding boxes with class ID and confidence score
4. WHEN the model is loaded, THE Detection_System SHALL log the model name, version, and parameter count
5. WHEN CUDA GPU is available, THE Detection_System SHALL automatically enable FP16 half-precision inference for speed
6. WHEN a frame is processed, THE Detection_System SHALL complete YOLOv26 inference within 100ms on standard CPU hardware
7. WHEN the model file is missing, THE Detection_System SHALL raise a FileNotFoundError with a clear message directing the user to download the model

### Requirement 2: Implement BoT-SORT Multi-Object Tracking

**User Story:** As a system operator, I want multi-object tracking to reliably track license plates across video frames, so that I can reduce duplicate detections and improve plate stabilization.

#### Acceptance Criteria

1. WHEN a video is processed, THE Tracking_System SHALL initialize a BoT-SORT tracker instance at the start of the video
2. WHEN YOLOv26 detections are available, THE Tracking_System SHALL pass detections to BoT-SORT with bounding boxes and confidence scores
3. WHEN BoT-SORT processes detections, THE Tracking_System SHALL assign unique track IDs to each detected plate
4. WHEN a plate is tracked across frames, THE Tracking_System SHALL maintain consistent track ID across ≥2 consecutive frames
5. WHEN a plate leaves the frame or is occluded, THE Tracking_System SHALL mark the track as inactive after 30 frames without detection
6. WHEN multiple plates are detected in a single frame, THE Tracking_System SHALL assign unique track IDs to each plate
7. WHEN BoT-SORT is initialized, THE Tracking_System SHALL log the tracker configuration (max age, min hits, iou threshold)

### Requirement 3: Upgrade OCR to PaddleOCR (PP-OCRv5)

**User Story:** As a system operator, I want OCR recognition to use PaddleOCR PP-OCRv5 instead of CRNN, so that I can achieve higher text recognition accuracy on Indian license plates.

#### Acceptance Criteria

1. WHEN the backend starts, THE OCR_System SHALL load the PaddleOCR PP-OCRv5 model
2. WHEN a plate crop is provided, THE OCR_System SHALL run PaddleOCR inference and return recognized text with confidence score
3. WHEN PaddleOCR processes a plate image, THE OCR_System SHALL return text with confidence ≥ 0.25 (or skip if below threshold)
4. WHEN OCR confidence is below 0.25, THE OCR_System SHALL return empty string and log the low confidence
5. WHEN multiple OCR variants are tested (original, CLAHE, sharpened, thresholded), THE OCR_System SHALL select the variant with highest confidence
6. WHEN PaddleOCR is unavailable, THE OCR_System SHALL fall back to CRNN recognizer (if weights available) or EasyOCR
7. WHEN the OCR model is loaded, THE OCR_System SHALL log the model name, version, and language support

### Requirement 4: Implement Grammar Validation for Indian License Plates

**User Story:** As a system operator, I want OCR results to be validated against Indian license plate format rules, so that I can reject garbage recognition results and improve accuracy.

#### Acceptance Criteria

1. WHEN OCR returns text, THE Grammar_Validator SHALL validate the text against Indian RTO plate format (AA NN AA NNNN)
2. WHEN text matches the format, THE Grammar_Validator SHALL return validation status "VALID"
3. WHEN text does not match the format, THE Grammar_Validator SHALL return validation status "INVALID" with reason
4. WHEN text contains character manipulation (e.g., "1" instead of "I"), THE Grammar_Validator SHALL apply position-based corrections
5. WHEN text is in positions 0-1 (state code), THE Grammar_Validator SHALL enforce letters only (convert digits to letters if possible)
6. WHEN text is in positions 2-3 (district code), THE Grammar_Validator SHALL enforce digits only (convert letters to digits if possible)
7. WHEN text is in positions 4+ (series + registration), THE Grammar_Validator SHALL enforce series as letters and registration as digits
8. WHEN text length is < 6 or > 12 characters, THE Grammar_Validator SHALL reject as invalid

### Requirement 5: Implement Levenshtein Distance Correction

**User Story:** As a system operator, I want similar plate detections to be deduplicated using string similarity metrics, so that I can reduce false positives from OCR variations.

#### Acceptance Criteria

1. WHEN multiple plates are detected in a video, THE Deduplicator SHALL calculate Levenshtein distance between normalized plate strings
2. WHEN Levenshtein distance ≤ 2 AND length difference ≤ 2, THE Deduplicator SHALL consider plates as duplicates
3. WHEN duplicates are found, THE Deduplicator SHALL keep the detection with highest combined confidence
4. WHEN a duplicate is skipped, THE Deduplicator SHALL log the original and duplicate plate with distance metric
5. WHEN Levenshtein distance > 2, THE Deduplicator SHALL treat plates as distinct detections
6. WHEN deduplication is complete, THE Deduplicator SHALL return a list of unique plates with highest confidence readings

### Requirement 6: Maintain Backward Compatibility with Frontend

**User Story:** As a frontend developer, I want the backend API to maintain the same response format, so that I can use the existing HTML/CSS/JS dashboard without modification.

#### Acceptance Criteria

1. WHEN the frontend calls `/analyze-video`, THE Backend SHALL return JSON with same structure: `{"detections": [...]}`
2. WHEN the frontend calls `/api/live-detections`, THE Backend SHALL return JSON with same structure: `{"detections": [...]}`
3. WHEN the frontend calls `/api/cameras/{camera_id}`, THE Backend SHALL return JSON with same structure: `{"camera": {...}}`
4. WHEN the frontend calls `/api/detections/{camera_id}`, THE Backend SHALL return JSON with same structure: `{"detections": [...]}`
5. WHEN the frontend displays detection results, THE Backend SHALL include `detected_plate`, `correct_plate`, `violation`, `confidence`, `bbox`, and `plate_image` fields
6. WHEN the frontend sends a frame for live monitoring, THE Backend SHALL accept the same multipart form data format
7. WHEN the backend processes a frame, THE Backend SHALL return bounding boxes as percentages (0-100) for canvas overlay

### Requirement 7: Improve Detection Accuracy with Enhanced Preprocessing

**User Story:** As a system operator, I want plate crops to be preprocessed with advanced image enhancement, so that I can improve OCR accuracy on low-quality video frames.

#### Acceptance Criteria

1. WHEN a plate crop is extracted, THE Preprocessor SHALL apply bilateral filtering to reduce noise while preserving edges
2. WHEN a plate crop is processed, THE Preprocessor SHALL apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle shadows and glare
3. WHEN a plate crop is enhanced, THE Preprocessor SHALL apply sharpening kernel to enhance character edges
4. WHEN a plate crop is thresholded, THE Preprocessor SHALL test multiple threshold methods (Gaussian, Mean, OTSU) and select the best
5. WHEN a plate crop is smaller than 320×120, THE Preprocessor SHALL upscale using cubic interpolation
6. WHEN preprocessing is complete, THE Preprocessor SHALL return a 320×120 binary image ready for OCR
7. WHEN multiple preprocessing variants are generated, THE Preprocessor SHALL return 4 variants: original, CLAHE-only, sharpened, thresholded

### Requirement 8: Implement Plate Stabilization Across Frames

**User Story:** As a system operator, I want detected plates to be confirmed across multiple frames before logging, so that I can reduce false positives from transient OCR errors.

#### Acceptance Criteria

1. WHEN a plate is detected in a frame, THE Stabilizer SHALL track the normalized plate string
2. WHEN the same plate is detected in ≥2 consecutive frames, THE Stabilizer SHALL mark the plate as "confirmed"
3. WHEN a plate is confirmed, THE Stabilizer SHALL save the detection to the database with highest confidence reading
4. WHEN a plate is not confirmed after 30 seconds, THE Stabilizer SHALL expire the tracker entry
5. WHEN a plate is detected in only 1 frame, THE Stabilizer SHALL NOT save to database (wait for confirmation)
6. WHEN a confirmed plate is displayed on the frontend, THE Stabilizer SHALL include `frames_seen` field showing how many frames the plate was detected in
7. WHEN a plate is first detected, THE Stabilizer SHALL show "Detecting..." on the frontend until confirmed

### Requirement 9: Enhance Combined Confidence Scoring

**User Story:** As a system operator, I want detection confidence to combine YOLO and OCR scores with format validation boost, so that I can prioritize high-confidence detections.

#### Acceptance Criteria

1. WHEN a plate is detected, THE Confidence_Scorer SHALL calculate base confidence as: (YOLO_confidence × 0.4) + (OCR_confidence × 0.6)
2. WHEN a plate matches Indian RTO format, THE Confidence_Scorer SHALL apply format boost of 1.15× to base confidence
3. WHEN a plate is confirmed across multiple frames, THE Confidence_Scorer SHALL apply stability boost of +0.05 per additional frame (max 1.0)
4. WHEN confidence is calculated, THE Confidence_Scorer SHALL cap final score at 1.0
5. WHEN confidence is below 0.50, THE Confidence_Scorer SHALL mark detection as low-confidence
6. WHEN confidence is ≥ 0.50, THE Confidence_Scorer SHALL mark detection as high-confidence
7. WHEN confidence is calculated, THE Confidence_Scorer SHALL round to 2 decimal places

### Requirement 10: Support Video Processing with Frame Sampling

**User Story:** As a system operator, I want to process videos efficiently by sampling frames, so that I can analyze long videos without excessive computation.

#### Acceptance Criteria

1. WHEN a video is uploaded, THE Video_Processor SHALL extract frames at configurable interval (default: every 3rd frame)
2. WHEN frames are extracted, THE Video_Processor SHALL pass each frame to the detection pipeline
3. WHEN a frame is processed, THE Video_Processor SHALL record the frame number with each detection
4. WHEN video processing is complete, THE Video_Processor SHALL return all detections with frame numbers
5. WHEN a video is invalid or corrupted, THE Video_Processor SHALL return empty detection list and log error
6. WHEN frame extraction fails, THE Video_Processor SHALL continue processing remaining frames
7. WHEN video processing completes, THE Video_Processor SHALL log total frames processed and detections found

### Requirement 11: Support Live Camera Feed Processing

**User Story:** As a system operator, I want to process live camera frames in real-time, so that I can monitor traffic and detect violations as they occur.

#### Acceptance Criteria

1. WHEN a camera frame is uploaded to `/api/process-frame`, THE Frame_Processor SHALL detect plates using YOLOv26
2. WHEN a frame is processed, THE Frame_Processor SHALL return bounding boxes as percentages (0-100) for canvas overlay
3. WHEN a plate is detected, THE Frame_Processor SHALL run OCR and return recognized text
4. WHEN OCR succeeds, THE Frame_Processor SHALL validate the plate and return violation status
5. WHEN a plate is first detected, THE Frame_Processor SHALL show "Detecting..." until confirmed in ≥2 frames
6. WHEN a plate is confirmed, THE Frame_Processor SHALL return full detection data with plate text and violation status
7. WHEN frame dimensions are provided, THE Frame_Processor SHALL scale bounding boxes to match original resolution

### Requirement 12: Maintain SQLite Database Persistence

**User Story:** As a system operator, I want all detections to be persisted to SQLite, so that I can query historical records and generate reports.

#### Acceptance Criteria

1. WHEN a detection is confirmed, THE Database SHALL insert a record with plate text, confidence, frame number, and timestamp
2. WHEN a detection is inserted, THE Database SHALL include camera ID, violation status, and vehicle registration info (if available)
3. WHEN the database is queried, THE Database SHALL return detections sorted by timestamp (newest first)
4. WHEN a camera is created, THE Database SHALL insert a camera record with name, location, and resolution
5. WHEN camera info is updated, THE Database SHALL update the camera record with new metadata
6. WHEN detections are queried, THE Database SHALL support filtering by camera ID and violation status
7. WHEN the database is initialized, THE Database SHALL create tables if they don't exist

### Requirement 13: Implement Fallback OCR Pipeline

**User Story:** As a system operator, I want OCR to gracefully fall back to alternative engines if PaddleOCR fails, so that the system remains operational.

#### Acceptance Criteria

1. WHEN PaddleOCR is unavailable, THE OCR_System SHALL fall back to CRNN recognizer (if weights available)
2. WHEN CRNN is unavailable, THE OCR_System SHALL fall back to EasyOCR
3. WHEN EasyOCR is unavailable, THE OCR_System SHALL fall back to Tesseract
4. WHEN all OCR engines are unavailable, THE OCR_System SHALL log error and return empty string
5. WHEN a fallback is used, THE OCR_System SHALL log which engine was used
6. WHEN the backend starts, THE OCR_System SHALL check availability of all OCR engines and log status

### Requirement 14: Log Detection Pipeline Events

**User Story:** As a system operator, I want detailed logs of the detection pipeline, so that I can debug issues and monitor system performance.

#### Acceptance Criteria

1. WHEN the backend starts, THE Logger SHALL log model loading status (YOLO, OCR, tracking)
2. WHEN a video is processed, THE Logger SHALL log frame count and detection count
3. WHEN a plate is detected, THE Logger SHALL log plate text, confidence, and violation status
4. WHEN a plate is stabilized, THE Logger SHALL log frame count and final confidence
5. WHEN a plate is deduplicated, THE Logger SHALL log original and duplicate plate with Levenshtein distance
6. WHEN an error occurs, THE Logger SHALL log the error with stack trace
7. WHEN the pipeline completes, THE Logger SHALL log total processing time and results summary

### Requirement 15: Parse and Pretty-Print License Plates

**User Story:** As a system operator, I want license plates to be parsed into structured components and formatted consistently, so that I can display them correctly and validate format.

#### Acceptance Criteria

1. WHEN a plate string is parsed, THE Parser SHALL extract state code (2 letters), district code (2 digits), series (1-3 letters), and registration number (1-4 digits)
2. WHEN a plate is parsed, THE Parser SHALL return a structured object with separate fields for each component
3. WHEN a plate is pretty-printed, THE Pretty_Printer SHALL format as "AA NN AA NNNN" with spaces between components
4. WHEN a plate is parsed then pretty-printed then parsed again, THE Round_Trip_Property SHALL produce an equivalent object (parse → print → parse)
5. WHEN a plate contains character manipulation, THE Parser SHALL apply corrections before parsing
6. WHEN a plate is invalid, THE Parser SHALL return error with reason
7. WHEN a plate is pretty-printed, THE Pretty_Printer SHALL return a string matching Indian RTO format

### Requirement 16: Handle Character Manipulation Detection

**User Story:** As a system operator, I want to detect and flag plates with character manipulation (e.g., "1" instead of "I"), so that I can identify illegal modifications.

#### Acceptance Criteria

1. WHEN a plate is validated, THE Validator SHALL check for common character substitutions (0→O, 1→I, 8→B, etc.)
2. WHEN character manipulation is detected, THE Validator SHALL flag the plate with "Character_Manipulation" violation
3. WHEN a plate has character manipulation, THE Validator SHALL apply corrections and return both original and corrected text
4. WHEN a corrected plate matches Indian format, THE Validator SHALL return the corrected plate as `correct_plate`
5. WHEN a plate cannot be corrected, THE Validator SHALL return violation status "Character_Manipulation"
6. WHEN character manipulation is detected, THE Validator SHALL log the original and corrected plate

### Requirement 17: Detect Font Anomalies

**User Story:** As a system operator, I want to detect plates with non-standard fonts or anomalies, so that I can flag suspicious plates for manual review.

#### Acceptance Criteria

1. WHEN a plate image is analyzed, THE Font_Analyzer SHALL check for font consistency across characters
2. WHEN font anomalies are detected, THE Font_Analyzer SHALL flag the plate with "Font_Anomaly" violation
3. WHEN a plate has font anomalies, THE Font_Analyzer SHALL return `font_anomaly: true` in the detection result
4. WHEN a plate has standard font, THE Font_Analyzer SHALL return `font_anomaly: false`
5. WHEN font analysis fails, THE Font_Analyzer SHALL return `font_anomaly: null` (unknown)

### Requirement 18: Support Vehicle Registration Lookup

**User Story:** As a system operator, I want to look up vehicle registration information from detected plates, so that I can identify registered vs. unregistered vehicles.

#### Acceptance Criteria

1. WHEN a plate is detected, THE Registration_Lookup SHALL query the SQLite database for matching registration
2. WHEN a registration is found, THE Registration_Lookup SHALL return vehicle info (owner, registration date, status)
3. WHEN a registration is not found, THE Registration_Lookup SHALL return `vehicle_info: null`
4. WHEN a plate is unregistered, THE Validator SHALL flag with "Unregistered_Vehicle" violation
5. WHEN a plate is registered, THE Validator SHALL return `vehicle_info` with registration details
6. WHEN vehicle info is returned, THE Detection_Result SHALL include `vehicle_info` field in JSON response

### Requirement 19: Validate Against RTO Rules

**User Story:** As a system operator, I want detected plates to be validated against Indian RTO rules, so that I can identify illegal plates and violations.

#### Acceptance Criteria

1. WHEN a plate is detected, THE RTO_Validator SHALL check against Indian license plate format rules
2. WHEN a plate matches RTO format, THE RTO_Validator SHALL return `violation: null`
3. WHEN a plate does not match RTO format, THE RTO_Validator SHALL return violation type (e.g., "Character_Manipulation", "Invalid_Format")
4. WHEN a plate is unregistered, THE RTO_Validator SHALL return `violation: "Unregistered_Vehicle"`
5. WHEN a plate has character manipulation, THE RTO_Validator SHALL return `violation: "Character_Manipulation"`
6. WHEN a plate has font anomalies, THE RTO_Validator SHALL return `violation: "Font_Anomaly"`
7. WHEN a plate is valid, THE RTO_Validator SHALL return `violation: null` and `correct_plate: null`

### Requirement 20: Encode Plate Images as Base64

**User Story:** As a frontend developer, I want plate crop images to be encoded as base64 data URLs, so that I can display them directly in the HTML without additional requests.

#### Acceptance Criteria

1. WHEN a plate is detected, THE Image_Encoder SHALL encode the plate crop as JPEG
2. WHEN a plate crop is encoded, THE Image_Encoder SHALL return a data URL: `data:image/jpeg;base64,...`
3. WHEN encoding fails, THE Image_Encoder SHALL return `null` and log warning
4. WHEN a plate image is encoded, THE Image_Encoder SHALL use JPEG quality 85 for balance between size and quality
5. WHEN a detection is returned to frontend, THE Detection_Result SHALL include `plate_image` field with base64 data URL

---

## Acceptance Criteria Testing Strategy

### Property-Based Testing Approach

The following acceptance criteria will be tested using property-based testing (PBT) to ensure correctness across a wide range of inputs:

#### 1. Round-Trip Property: Parse → Pretty-Print → Parse (Requirement 15)
- **Property**: FOR ALL valid license plates, parsing then pretty-printing then parsing SHALL produce an equivalent object
- **Test**: Generate random valid Indian plates, parse → print → parse, verify equivalence
- **Tool**: Hypothesis (Python)

#### 2. Invariant Property: Plate Stabilization (Requirement 8)
- **Property**: WHEN a plate is confirmed across ≥2 frames, the track ID SHALL remain constant
- **Test**: Generate random frame sequences with repeated plates, verify track ID consistency
- **Tool**: Hypothesis (Python)

#### 3. Metamorphic Property: Levenshtein Distance (Requirement 5)
- **Property**: Levenshtein distance between two strings SHALL be symmetric: distance(A, B) = distance(B, A)
- **Test**: Generate random plate strings, verify distance symmetry
- **Tool**: Hypothesis (Python)

#### 4. Idempotence Property: Deduplication (Requirement 5)
- **Property**: Running deduplication twice on the same detection list SHALL produce the same result
- **Test**: Generate random detection lists, deduplicate twice, verify idempotence
- **Tool**: Hypothesis (Python)

#### 5. Confluence Property: Character Correction (Requirement 4)
- **Property**: Applying position-based corrections in any order SHALL produce the same result
- **Test**: Generate random plates with character errors, apply corrections in different orders, verify confluence
- **Tool**: Hypothesis (Python)

#### 6. Error Condition Property: Invalid Plates (Requirement 4)
- **Property**: WHEN a plate is invalid, the validator SHALL return error status with reason
- **Test**: Generate invalid plates (wrong length, wrong format, garbage), verify error handling
- **Tool**: Hypothesis (Python)

#### 7. Confidence Scoring Property (Requirement 9)
- **Property**: Combined confidence SHALL be in range [0.0, 1.0] and SHALL increase with format match
- **Test**: Generate random YOLO/OCR confidence pairs, verify combined score is valid and monotonic
- **Tool**: Hypothesis (Python)

#### 8. Bounding Box Scaling Property (Requirement 11)
- **Property**: Bounding boxes scaled from processing resolution to display resolution SHALL be within frame bounds
- **Test**: Generate random frame sizes and bounding boxes, verify scaled boxes are within bounds
- **Tool**: Hypothesis (Python)

