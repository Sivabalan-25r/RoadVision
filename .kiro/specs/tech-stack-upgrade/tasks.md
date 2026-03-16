# Implementation Plan: RoadVision Tech Stack Upgrade

## Overview

This implementation plan breaks down the tech stack upgrade into discrete, incremental coding tasks. The upgrade replaces YOLOv8n + CRNN with YOLOv26 + BoT-SORT + PaddleOCR, adding grammar validation, Levenshtein deduplication, and enhanced confidence scoring. All tasks maintain backward compatibility with the existing frontend.

The implementation follows a logical progression: detection → tracking → OCR → validation → integration → testing, ensuring each component is validated before moving to the next phase.

---

## Phase 1: Detection Module Upgrade (YOLOv26)

- [x] 1. Set up YOLOv26 model loading and configuration
  - Create `backend-python/config.py` with all model paths and thresholds
  - Update `backend-python/recognition/plate_reader.py` to load YOLOv26 instead of YOLOv8n
  - Add FP16 half-precision auto-detection for CUDA GPU
  - Add startup logging for model parameters and inference time
  - _Requirements: 1.1, 1.2, 1.4, 1.5_

  - [x] 1.1 Write property test for YOLOv26 detection output format
    - **Property 1: YOLOv26 Detection Returns Required Fields**
    - **Validates: Requirements 1.2, 1.3**

- [x] 2. Implement geometric filtering for plate detection
  - Add aspect ratio validation (1.5–7.0 range)
  - Add area constraints (750–100,000 pixels)
  - Add frame area ratio check (reject if >15% of frame)
  - Update `detect_plates()` to apply filters before returning detections
  - _Requirements: 1.3, 7.1_

- [x] 3. Implement bounding box refinement
  - Add light padding removal (2% from edges) to remove vehicle body
  - Ensure refined crops don't cut off plate characters
  - Update detection pipeline to use refined crops for OCR
  - _Requirements: 7.1_

- [x] 4. Checkpoint - Verify YOLOv26 detection works
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 2: Tracking Module (BoT-SORT)

- [ ] 5. Implement BoT-SORT multi-object tracking
  - Install `bot-sort` package (or equivalent tracking library)
  - Create `backend-python/tracking/botsort_tracker.py` with BoT-SORT wrapper
  - Initialize tracker with configuration: max_age=30, min_hits=1, iou_threshold=0.1
  - Implement `update()` method to assign track IDs to detections
  - Add logging for tracker initialization and track assignments
  - _Requirements: 2.1, 2.2, 2.3, 2.7_

  - [ ]* 5.1 Write property test for BoT-SORT track ID uniqueness
    - **Property 2: BoT-SORT Assigns Unique Track IDs**
    - **Validates: Requirements 2.3, 2.6**

  - [ ]* 5.2 Write property test for track ID consistency across frames
    - **Property 3: Track ID Consistency Across Frames**
    - **Validates: Requirements 2.4**

  - [ ]* 5.3 Write property test for track expiry after inactivity
    - **Property 4: Track Expiry After Inactivity**
    - **Validates: Requirements 2.5**

- [ ] 6. Integrate BoT-SORT into video processing pipeline
  - Update `backend-python/processing/video_processor.py` to initialize tracker at video start
  - Pass YOLO detections to BoT-SORT for track ID assignment
  - Include track IDs in detection results
  - _Requirements: 2.1, 2.2_

- [ ] 7. Checkpoint - Verify BoT-SORT tracking works
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 3: OCR Module Upgrade (PaddleOCR)

- [ ] 8. Implement PaddleOCR PP-OCRv5 integration
  - Update `backend-python/recognition/plate_reader.py` to load PaddleOCR model
  - Create `recognize_plate_paddleocr()` function for PaddleOCR inference
  - Add confidence threshold check (0.25 minimum)
  - Add logging for model loading and inference time
  - _Requirements: 3.1, 3.2, 3.3, 3.7_

  - [ ]* 8.1 Write property test for OCR output format
    - **Property 5: OCR Returns Text and Confidence**
    - **Validates: Requirements 3.2**

  - [ ]* 8.2 Write property test for OCR confidence thresholding
    - **Property 6: OCR Confidence Thresholding**
    - **Validates: Requirements 3.3, 3.4**

- [ ] 9. Implement OCR fallback chain
  - Create fallback sequence: PaddleOCR → CRNN → EasyOCR → Tesseract
  - Update `read_plate()` to try each engine in sequence
  - Add logging for fallback usage
  - Handle missing dependencies gracefully
  - _Requirements: 3.6, 13.1, 13.2, 13.3, 13.4, 13.5_

- [ ] 10. Implement ensemble OCR with preprocessing variants
  - Create `preprocess_plate_variants()` to generate 4 variants: original, CLAHE, sharpened, thresholded
  - Run OCR on each variant
  - Select variant with highest confidence
  - Add logging for ensemble selection
  - _Requirements: 3.5, 7.2, 7.3, 7.4, 7.7_

  - [ ]* 10.1 Write property test for ensemble OCR selection
    - **Property 7: Ensemble OCR Selects Highest Confidence**
    - **Validates: Requirements 3.5**

- [ ] 11. Implement advanced plate preprocessing
  - Add bilateral filtering for noise reduction
  - Add CLAHE for shadow/glare handling
  - Add sharpening kernel for character edge enhancement
  - Add adaptive thresholding (Gaussian, Mean, OTSU)
  - Ensure output is 320×120 pixels with cubic interpolation upscaling
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 11.1 Write property test for preprocessing output dimensions
    - **Property 18: Preprocessing Output Dimensions**
    - **Validates: Requirements 7.6**

  - [ ]* 11.2 Write property test for preprocessing variant generation
    - **Property 19: Preprocessing Generates Four Variants**
    - **Validates: Requirements 7.7**

- [ ] 12. Checkpoint - Verify PaddleOCR integration works
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 4: Validation Module (Grammar & Character Correction)

- [ ] 13. Implement Indian RTO format validation
  - Create `backend-python/rules/grammar_validator.py`
  - Implement `validate_indian_format()` to check AA NN AA NNNN pattern
  - Add length validation (6–12 characters)
  - Add logging for validation results
  - _Requirements: 4.1, 4.2, 4.3, 4.8_

  - [ ]* 13.1 Write property test for grammar validation
    - **Property 8: Grammar Validation Against Indian Format**
    - **Validates: Requirements 4.1**

- [ ] 14. Implement position-based character corrections
  - Create `apply_position_based_corrections()` function
  - Positions 0-1 (state code): enforce letters (0→O, 1→I, 8→B, etc.)
  - Positions 2-3 (district code): enforce digits (O→0, I→1, B→8, etc.)
  - Positions 4+ (series + registration): enforce series as letters, registration as digits
  - Add logging for corrections applied
  - _Requirements: 4.4, 4.5, 4.6, 4.7_

  - [ ]* 14.1 Write property test for position-based corrections
    - **Property 9: Position-Based Character Corrections**
    - **Validates: Requirements 4.4, 4.5, 4.6, 4.7**

- [ ] 15. Implement character manipulation detection
  - Create `detect_character_manipulation()` function
  - Check for common substitutions (0→O, 1→I, 8→B, 5→S, 2→Z, 6→G)
  - Flag plates with manipulation as violation
  - Return both original and corrected text
  - _Requirements: 16.1, 16.2, 16.3, 16.4_

  - [ ]* 15.1 Write property test for character manipulation detection
    - **Property 42: Character Manipulation Detection**
    - **Validates: Requirements 16.1, 16.2**

  - [ ]* 15.2 Write property test for character manipulation correction
    - **Property 43: Character Manipulation Correction**
    - **Validates: Requirements 16.3, 16.4**

- [ ] 16. Implement font anomaly detection
  - Create `detect_font_anomalies()` function
  - Analyze plate image for font consistency
  - Return boolean flag: true (anomalies), false (standard), null (unknown)
  - Add logging for anomaly detection
  - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5_

  - [ ]* 16.1 Write property test for font anomaly detection output
    - **Property 44: Font Anomaly Detection Output**
    - **Validates: Requirements 17.3, 17.4, 17.5**

- [ ] 17. Implement vehicle registration lookup
  - Create `lookup_vehicle_registration()` function
  - Query SQLite `vehicle_registrations` table by plate
  - Return vehicle info (owner, registration_date, status) or null
  - Add logging for lookup results
  - _Requirements: 18.1, 18.2, 18.3_

  - [ ]* 17.1 Write property test for vehicle registration lookup
    - **Property 45: Vehicle Registration Lookup**
    - **Validates: Requirements 18.1, 18.2, 18.3**

- [ ] 18. Implement comprehensive plate validation
  - Create `validate_plate()` function that combines all validators
  - Check grammar, character manipulation, font anomalies, registration
  - Return `ValidationResult` with detected_plate, correct_plate, violation, confidence_modifier
  - Add logging for validation results
  - _Requirements: 4.1, 16.1, 17.1, 18.4, 19.1, 19.2, 19.3_

  - [ ]* 18.1 Write property test for RTO format validation
    - **Property 48: RTO Format Validation**
    - **Validates: Requirements 19.1, 19.2, 19.3**

  - [ ]* 18.2 Write property test for violation type detection
    - **Property 49: RTO Violation Types**
    - **Validates: Requirements 19.4, 19.5, 19.6, 19.7**

- [ ] 19. Checkpoint - Verify validation pipeline works
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 5: Deduplication Module (Levenshtein Distance)

- [ ] 20. Implement Levenshtein distance calculation
  - Create `backend-python/deduplication/levenshtein.py`
  - Implement `levenshtein_distance()` function
  - Add memoization for performance
  - Add logging for distance calculations
  - _Requirements: 5.1_

  - [ ]* 20.1 Write property test for Levenshtein distance symmetry
    - **Property 11: Levenshtein Distance Symmetry**
    - **Validates: Requirements 5.1**

- [ ] 21. Implement plate deduplication logic
  - Create `deduplicate_detections()` function
  - Group detections by normalized plate string
  - Calculate Levenshtein distance between groups
  - Merge groups with distance ≤2 AND length difference ≤2
  - Keep detection with highest confidence
  - Add logging for duplicates found and merged
  - _Requirements: 5.2, 5.3, 5.5, 5.6_

  - [ ]* 21.1 Write property test for duplicate detection
    - **Property 12: Duplicate Detection by Levenshtein Distance**
    - **Validates: Requirements 5.2**

  - [ ]* 21.2 Write property test for distinct plates not merged
    - **Property 14: Distinct Plates Not Merged**
    - **Validates: Requirements 5.5**

  - [ ]* 21.3 Write property test for deduplication idempotence
    - **Property 13: Deduplication Selects Highest Confidence**
    - **Validates: Requirements 5.3, 5.6**

- [ ] 22. Checkpoint - Verify deduplication works
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 6: Confidence Scoring & Stabilization

- [ ] 23. Implement combined confidence scoring
  - Create `backend-python/scoring/confidence_scorer.py`
  - Implement `calculate_confidence()` function
  - Formula: (YOLO_conf × 0.4 + OCR_conf × 0.6) × format_boost × stability_boost
  - Apply format boost (1.15×) for valid Indian format plates
  - Cap final score at 1.0, round to 2 decimals
  - Add logging for confidence calculations
  - _Requirements: 9.1, 9.2, 9.4, 9.7_

  - [ ]* 23.1 Write property test for confidence score calculation
    - **Property 24: Confidence Score Calculation**
    - **Validates: Requirements 9.1, 9.4**

  - [ ]* 23.2 Write property test for format boost application
    - **Property 25: Format Boost Applied to Valid Plates**
    - **Validates: Requirements 9.2**

  - [ ]* 23.3 Write property test for stability boost
    - **Property 26: Stability Boost for Multi-Frame Detections**
    - **Validates: Requirements 9.3**

- [ ] 24. Implement plate stabilization tracker
  - Create `backend-python/stabilization/plate_stabilizer.py`
  - Implement `stabilize_detection()` function
  - Track normalized plates across frames
  - Require ≥2 frames for confirmation
  - Expire entries after 30 seconds
  - Return confirmation status (True/False)
  - Add logging for stabilization events
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 24.1 Write property test for stabilization multi-frame requirement
    - **Property 20: Plate Stabilization Requires Multiple Frames**
    - **Validates: Requirements 8.2, 8.5**

  - [ ]* 24.2 Write property test for stabilization expiry
    - **Property 21: Stabilization Expiry After Timeout**
    - **Validates: Requirements 8.4**

- [ ] 25. Integrate stabilization into frame processor
  - Update `/api/process-frame` endpoint to use stabilizer
  - Return "Detecting..." for unconfirmed plates
  - Return full data for confirmed plates
  - Include `frames_seen` field in response
  - _Requirements: 8.6, 8.7, 11.5_

  - [ ]* 25.1 Write property test for unconfirmed plate display
    - **Property 23: Unconfirmed Plates Show "Detecting..."**
    - **Validates: Requirements 8.7, 11.5**

  - [ ]* 25.2 Write property test for confirmed plate frames_seen field
    - **Property 22: Confirmed Plates Include frames_seen Field**
    - **Validates: Requirements 8.6**

- [ ] 26. Checkpoint - Verify confidence and stabilization work
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 7: Plate Parsing & Pretty-Printing

- [ ] 27. Implement license plate parser
  - Create `backend-python/parsing/plate_parser.py`
  - Implement `parse_plate()` function
  - Extract state code (2 letters), district code (2 digits), series (1-3 letters), registration (1-4 digits)
  - Return structured object with separate fields
  - Add validation for each component
  - _Requirements: 15.1, 15.2_

  - [ ]* 27.1 Write property test for plate parsing
    - **Property 39: Plate Parsing Extracts Components**
    - **Validates: Requirements 15.1, 15.2**

- [ ] 28. Implement license plate pretty-printer
  - Create `pretty_print_plate()` function
  - Format as "AA NN AA NNNN" with spaces between components
  - Ensure output matches Indian RTO format
  - _Requirements: 15.3, 15.7_

  - [ ]* 28.1 Write property test for plate pretty-printing
    - **Property 40: Plate Pretty-Printing Format**
    - **Validates: Requirements 15.3, 15.7**

  - [ ]* 28.2 Write property test for round-trip parsing
    - **Property 41: Plate Parsing Round-Trip**
    - **Validates: Requirements 15.4**

- [ ] 29. Checkpoint - Verify plate parsing works
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 8: Database Schema & API Updates

- [ ] 30. Update database schema for new fields
  - Add `track_id` column to `detections` table
  - Add `frames_seen` column to `detections` table
  - Add `font_anomaly` column to `detections` table
  - Add `vehicle_info` column (JSON) to `detections` table
  - Create indexes on `camera_id`, `timestamp`, `violation`
  - Update `database.py` with new schema
  - _Requirements: 12.1, 12.2, 12.3, 12.6_

- [ ] 31. Update API response format for backward compatibility
  - Ensure `/analyze-video` returns same structure with new fields
  - Ensure `/api/process-frame` returns percentages for bounding boxes
  - Ensure `/api/detections/{camera_id}` includes all new fields
  - Add `plate_image` field as base64 data URL
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.7, 20.1, 20.2, 20.4, 20.5_

  - [ ]* 31.1 Write property test for API response format
    - **Property 15: API Response Contains Required Fields**
    - **Validates: Requirements 6.5**

  - [ ]* 31.2 Write property test for bounding box percentages
    - **Property 16: Bounding Boxes as Percentages**
    - **Validates: Requirements 6.7, 11.2**

- [ ] 32. Implement plate image encoding
  - Create `encode_plate_image()` function
  - Encode plate crop as JPEG with quality 85
  - Return data URL: `data:image/jpeg;base64,...`
  - Handle encoding errors gracefully
  - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5_

- [ ] 33. Update video processing pipeline
  - Integrate all components: detection → tracking → OCR → validation → deduplication → stabilization
  - Update `/analyze-video` endpoint to use new pipeline
  - Ensure frame numbers are recorded with detections
  - Ensure detections are saved to database
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ]* 33.1 Write property test for video frame extraction
    - **Property 28: Video Frame Extraction at Interval**
    - **Validates: Requirements 10.1**

  - [ ]* 33.2 Write property test for frame numbers recorded
    - **Property 29: Frame Numbers Recorded with Detections**
    - **Validates: Requirements 10.3, 10.4**

  - [ ]* 33.3 Write property test for invalid video handling
    - **Property 30: Invalid Video Returns Empty Detections**
    - **Validates: Requirements 10.5**

  - [ ]* 33.4 Write property test for frame error resilience
    - **Property 31: Video Processing Continues on Frame Errors**
    - **Validates: Requirements 10.6**

- [ ] 34. Update live frame processing pipeline
  - Integrate all components into `/api/process-frame` endpoint
  - Scale bounding boxes to original resolution
  - Return percentages for canvas overlay
  - Include OCR results and validation status
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.6, 11.7_

  - [ ]* 34.1 Write property test for frame processor percentages
    - **Property 32: Frame Processor Returns Percentages**
    - **Validates: Requirements 11.2**

  - [ ]* 34.2 Write property test for OCR integration in frame processing
    - **Property 33: OCR Integration in Frame Processing**
    - **Validates: Requirements 11.3**

  - [ ]* 34.3 Write property test for validation integration
    - **Property 34: Validation Integration in Frame Processing**
    - **Validates: Requirements 11.4**

  - [ ]* 34.4 Write property test for bounding box scaling
    - **Property 35: Bounding Box Scaling to Original Resolution**
    - **Validates: Requirements 11.7**

- [ ] 35. Checkpoint - Verify database and API updates work
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 9: Logging & Configuration

- [ ] 36. Implement comprehensive logging
  - Add logging for model loading (YOLO, OCR, tracking)
  - Add logging for video processing (frame count, detection count)
  - Add logging for plate detection (text, confidence, violation)
  - Add logging for plate stabilization (frame count, final confidence)
  - Add logging for deduplication (original, duplicate, distance)
  - Add logging for errors with stack traces
  - Add logging for pipeline completion (total time, results summary)
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_

- [ ] 37. Create configuration management system
  - Update `backend-python/config.py` with all configurable parameters
  - Add environment variable support for overrides
  - Document all configuration options
  - Add validation for configuration values
  - _Requirements: 1.5, 2.7, 3.7, 4.8, 5.2, 8.2, 9.1, 10.1, 13.1_

- [ ] 38. Checkpoint - Verify logging and configuration work
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 10: Integration Testing

- [ ] 39. Write end-to-end integration tests
  - Test complete video processing pipeline
  - Test live frame processing pipeline
  - Test database persistence
  - Test API response format
  - Test backward compatibility with frontend
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 39.1 Write integration test for video analysis endpoint
    - Test `/analyze-video` with sample video
    - Verify detections are returned with all required fields
    - Verify database records are created
    - _Requirements: 6.1, 6.5_

  - [ ]* 39.2 Write integration test for frame processing endpoint
    - Test `/api/process-frame` with sample frame
    - Verify bounding boxes are percentages
    - Verify OCR results are included
    - _Requirements: 6.6, 6.7, 11.2, 11.3_

  - [ ]* 39.3 Write integration test for camera endpoints
    - Test `/api/cameras` and `/api/cameras/{camera_id}`
    - Verify camera info is returned correctly
    - _Requirements: 6.2, 6.3_

  - [ ]* 39.4 Write integration test for detection history endpoint
    - Test `/api/detections/{camera_id}`
    - Verify filtering by violation status works
    - Verify sorting by timestamp works
    - _Requirements: 6.4, 12.3, 12.6_

- [ ] 40. Write error handling tests
  - Test missing model file handling
  - Test OCR fallback chain
  - Test invalid video handling
  - Test database error handling
  - Test API error responses
  - _Requirements: 1.7, 3.6, 10.5, 13.1, 13.4_

- [ ] 41. Checkpoint - Verify all integration tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 11: Performance & Optimization

- [ ] 42. Optimize detection performance
  - Profile YOLOv26 inference time
  - Verify FP16 half-precision is enabled on GPU
  - Optimize frame sampling (every 3rd frame)
  - Add performance logging
  - _Requirements: 1.6_

- [ ] 43. Optimize OCR performance
  - Profile PaddleOCR inference time
  - Verify ensemble processing is efficient
  - Add caching for OCR models
  - Add performance logging
  - _Requirements: 3.2_

- [ ] 44. Optimize database performance
  - Create indexes on frequently queried columns
  - Batch insert detections where possible
  - Profile query performance
  - Add performance logging
  - _Requirements: 12.3, 12.6_

- [ ] 45. Checkpoint - Verify performance meets requirements
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 12: Final Validation & Documentation

- [ ] 46. Verify backward compatibility with frontend
  - Test all API endpoints with existing frontend code
  - Verify response format matches expectations
  - Verify bounding box overlay works correctly
  - Verify plate image display works correctly
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [ ] 47. Verify all requirements are met
  - Cross-check each requirement against implementation
  - Verify all acceptance criteria are satisfied
  - Verify all properties are tested
  - _Requirements: All_

- [ ] 48. Final checkpoint - All systems operational
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional property-based tests and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation and early error detection
- Property tests validate universal correctness properties across all inputs
- Unit tests validate specific examples and edge cases
- All code must maintain backward compatibility with existing frontend
- Configuration should be externalized for easy deployment customization
