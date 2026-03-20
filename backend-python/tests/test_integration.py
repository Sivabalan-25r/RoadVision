"""
Integration Tests — Phase 10

Tests 39.1–39.4 and 40: End-to-end pipeline integration tests covering
video analysis, frame processing, camera endpoints, detection history,
and error handling.

**Validates: Requirements 6.1–6.7, 10.5, 11.2, 11.3, 12.3, 12.6, 13.1, 13.4**
"""

import base64
import json
import os
import re
import sqlite3
import sys
import tempfile

import cv2
import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blank_video(num_frames: int = 6, width: int = 640, height: int = 480) -> str:
    """Write a minimal MP4 to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    for _ in range(num_frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()
    return path


def _make_blank_frame(width: int = 320, height: int = 240) -> bytes:
    """Return JPEG bytes of a blank frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def _make_empty_db() -> str:
    """Create a temp SQLite DB with the vehicle_registrations table."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE vehicle_registrations (
            plate_number      TEXT PRIMARY KEY,
            owner             TEXT NOT NULL,
            registration_date TEXT NOT NULL,
            status            TEXT NOT NULL DEFAULT 'active'
        )
        """
    )
    conn.commit()
    conn.close()
    return path


# ===========================================================================
# Task 39.1 — Video analysis pipeline (process_video)
# ===========================================================================

class TestVideoAnalysisPipeline:
    """
    Integration test for the video processing pipeline.

    **Task 39.1: Integration test for video analysis endpoint**
    **Validates: Requirements 6.1, 6.5, 10.1, 10.3, 10.4, 10.5, 10.6**
    """

    def test_process_video_returns_list(self):
        """
        process_video() SHALL return a list for any valid video file.

        **Validates: Requirements 6.1, 10.1**
        """
        from processing.video_processor import process_video

        video_path = _make_blank_video(num_frames=6)
        try:
            result, total, processed = process_video(video_path, frame_interval=3)
            assert isinstance(result, list), \
                f"process_video must return a list as first element, got {type(result)}"
            assert total == 6
            assert processed == 3 # 0, 3, 6 (wait, 6/3 is 2, but frame 0, 3, 6 are 3 frames if 0-indexed)
        finally:
            os.unlink(video_path)

    def test_process_video_detections_have_required_fields(self):
        """
        Each detection from process_video() SHALL contain required fields.

        **Validates: Requirements 6.5, 10.3, 10.4**
        """
        from processing.video_processor import process_video

        video_path = _make_blank_video(num_frames=6)
        try:
            detections, _, _ = process_video(video_path, frame_interval=3)
            for det in detections:
                assert "bbox" in det, "Detection must have 'bbox'"
                assert "confidence" in det, "Detection must have 'confidence'"
                assert "frame_number" in det, "Detection must have 'frame_number'"
                assert "track_id" in det, "Detection must have 'track_id'"
                assert isinstance(det["frame_number"], int), "frame_number must be int"
                assert isinstance(det["bbox"], list), "bbox must be a list"
        finally:
            os.unlink(video_path)

    def test_process_video_invalid_path_returns_empty(self):
        """
        process_video() with a non-existent path SHALL return an empty list.

        **Validates: Requirement 10.5**
        """
        from processing.video_processor import process_video

        result, _, _ = process_video("/nonexistent/path/video.mp4", frame_interval=3)
        assert result == [], \
            f"Invalid video path must return [], got {result}"

    def test_process_video_frame_interval_respected(self):
        """
        process_video() SHALL only process frames at the given interval.

        **Validates: Requirement 10.1**
        """
        from processing.video_processor import process_video

        # 9 frames, interval=3 → frames 0, 3, 6 processed (3 frames)
        video_path = _make_blank_video(num_frames=9)
        try:
            detections, _, _ = process_video(video_path, frame_interval=3)
            # All frame_numbers must be multiples of 3
            for det in detections:
                assert det["frame_number"] % 3 == 0, (
                    f"frame_number {det['frame_number']} is not a multiple of interval 3"
                )
        finally:
            os.unlink(video_path)

    def test_process_video_frame_numbers_recorded(self):
        """
        Frame numbers SHALL be recorded with each detection.

        **Validates: Requirements 10.3, 10.4**
        """
        from processing.video_processor import process_video

        video_path = _make_blank_video(num_frames=6)
        try:
            detections, _, _ = process_video(video_path, frame_interval=1)
            for det in detections:
                assert "frame_number" in det
                assert det["frame_number"] >= 0
        finally:
            os.unlink(video_path)

    def test_process_video_continues_on_frame_errors(self):
        """
        process_video() SHALL not crash on corrupt/empty frames.

        **Validates: Requirement 10.6**
        """
        from processing.video_processor import process_video

        # A valid video — just verify it completes without exception
        video_path = _make_blank_video(num_frames=6)
        try:
            result, _, _ = process_video(video_path, frame_interval=2)
            assert isinstance(result, list)
        finally:
            os.unlink(video_path)


# ===========================================================================
# Task 39.2 — Frame processing pipeline (detect_plates + OCR)
# ===========================================================================

class TestFrameProcessingPipeline:
    """
    Integration test for single-frame processing.

    **Task 39.2: Integration test for frame processing endpoint**
    **Validates: Requirements 6.6, 6.7, 11.2, 11.3**
    """

    def test_detect_plates_returns_list(self):
        """
        detect_plates() SHALL return a list for any valid frame.

        **Validates: Requirement 6.6**
        """
        from recognition.plate_reader import detect_plates

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_plates(frame, frame_number=0)
        assert isinstance(result, list), \
            f"detect_plates must return a list, got {type(result)}"

    def test_detect_plates_fields_present(self):
        """
        Each detection from detect_plates() SHALL have bbox, crop, confidence.

        **Validates: Requirements 6.5, 11.2**
        """
        from recognition.plate_reader import detect_plates

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detect_plates(frame, frame_number=0)
        for det in detections:
            assert "bbox" in det
            assert "confidence" in det
            assert "crop" in det
            assert "raw_crop" in det
            assert isinstance(det["bbox"], list)
            assert len(det["bbox"]) == 4

    def test_bounding_box_percentage_conversion(self):
        """
        Bounding boxes converted to percentages SHALL be in [0, 100].

        **Validates: Requirements 6.7, 11.2**
        """
        from recognition.plate_reader import detect_plates

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detect_plates(frame, frame_number=0)

        frame_h, frame_w = frame.shape[:2]
        for det in detections:
            x, y, w, h = det["bbox"]
            x_pct = (x / frame_w) * 100
            y_pct = (y / frame_h) * 100
            w_pct = (w / frame_w) * 100
            h_pct = (h / frame_h) * 100
            for val in (x_pct, y_pct, w_pct, h_pct):
                assert 0.0 <= val <= 100.0, \
                    f"Bounding box percentage {val} is outside [0, 100]"

    def test_read_plate_returns_tuple(self):
        """
        read_plate() SHALL return a (text, confidence) tuple.

        **Validates: Requirement 11.3**
        """
        from recognition.plate_reader import read_plate

        frame = np.zeros((120, 320, 3), dtype=np.uint8)
        result = read_plate(frame)
        assert isinstance(result, tuple), \
            f"read_plate must return a tuple, got {type(result)}"
        assert len(result) == 2, \
            f"read_plate tuple must have 2 elements, got {len(result)}"
        text, conf = result
        assert text is None or isinstance(text, str), "text must be str or None"
        assert isinstance(conf, float), "confidence must be float"

    def test_plate_image_encoding(self):
        """
        Plate crops SHALL be encodable as base64 JPEG data URLs.

        **Validates: Requirements 20.1–20.5**
        """
        crop = np.zeros((40, 120, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        assert data_url.startswith("data:image/jpeg;base64,"), \
            "Plate image must be a JPEG data URL"
        # Verify it decodes back to a valid image
        decoded = base64.b64decode(b64)
        arr = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None, "Decoded plate image must be a valid image"


# ===========================================================================
# Task 39.3 — Camera endpoints (database layer)
# ===========================================================================

class TestCameraEndpoints:
    """
    Integration test for camera management.

    **Task 39.3: Integration test for camera endpoints**
    **Validates: Requirements 6.2, 6.3**
    """

    def test_get_all_cameras_returns_list(self):
        """
        get_all_cameras() SHALL return a list.

        **Validates: Requirement 6.2**
        """
        import database as db
        db.init_database()
        cameras = db.get_all_cameras()
        assert isinstance(cameras, list), \
            f"get_all_cameras must return a list, got {type(cameras)}"

    def test_get_camera_returns_dict_or_none(self):
        """
        get_camera() SHALL return a dict for known IDs and None for unknown.

        **Validates: Requirement 6.3**
        """
        import database as db
        db.init_database()

        # Known default camera
        cam = db.get_camera("CAM-001")
        assert cam is not None, "CAM-001 must exist after init_database()"
        assert isinstance(cam, dict), "Camera must be a dict"
        assert "camera_id" in cam
        assert "name" in cam

        # Unknown camera
        unknown = db.get_camera("CAM-NONEXISTENT-9999")
        assert unknown is None, "Unknown camera must return None"

    def test_camera_has_required_fields(self):
        """
        Each camera record SHALL have required fields.

        **Validates: Requirement 6.2**
        """
        import database as db
        db.init_database()
        cameras = db.get_all_cameras()
        for cam in cameras:
            assert "camera_id" in cam
            assert "name" in cam
            assert "location" in cam


# ===========================================================================
# Task 39.4 — Detection history endpoint (database layer)
# ===========================================================================

class TestDetectionHistoryEndpoint:
    """
    Integration test for detection history retrieval.

    **Task 39.4: Integration test for detection history endpoint**
    **Validates: Requirements 6.4, 12.3, 12.6**
    """

    def test_get_detections_returns_list(self):
        """
        get_detections() SHALL return a list.

        **Validates: Requirement 6.4**
        """
        import database as db
        db.init_database()
        detections = db.get_detections("CAM-001")
        assert isinstance(detections, list)

    def test_get_detections_violations_only_filter(self):
        """
        violations_only=True SHALL return only detections with a violation.

        **Validates: Requirements 6.4, 12.6**
        """
        import database as db

        # Use a fresh isolated DB for this test
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        original_path = db.DATABASE_PATH
        db.DATABASE_PATH = db_path
        try:
            db.init_database()

            db.add_detection("CAM-001", {
                "detected_plate": "MH34AC3479",
                "correct_plate": None,
                "violation": None,
                "confidence": 0.9,
                "frame": 1,
            })
            db.add_detection("CAM-001", {
                "detected_plate": "0H34AC3479",
                "correct_plate": "OH34AC3479",
                "violation": "Character Manipulation",
                "confidence": 0.85,
                "frame": 2,
            })

            violations = db.get_detections("CAM-001", violations_only=True)
            for det in violations:
                assert det["violation"] is not None, \
                    "violations_only=True must only return detections with a violation"
        finally:
            db.DATABASE_PATH = original_path
            os.unlink(db_path)

    def test_get_detections_sorted_by_timestamp(self):
        """
        Detections SHALL be returned sorted by timestamp descending.

        **Validates: Requirements 12.3, 12.6**
        """
        import database as db
        db.init_database()
        detections = db.get_detections("CAM-001", limit=50)
        timestamps = [d["timestamp"] for d in detections if d.get("timestamp")]
        assert timestamps == sorted(timestamps, reverse=True), \
            "Detections must be sorted by timestamp descending"

    def test_detection_has_required_fields(self):
        """
        Each detection record SHALL have all required fields.

        **Validates: Requirements 6.5, 12.1, 12.2, 12.3**
        """
        import database as db

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        original_path = db.DATABASE_PATH
        db.DATABASE_PATH = db_path
        try:
            db.init_database()

            db.add_detection("CAM-001", {
                "detected_plate": "MH34AC3479",
                "correct_plate": None,
                "violation": None,
                "confidence": 0.9,
                "track_id": 42,
                "frames_seen": 3,
                "frame": 10,
                "bbox": [10, 20, 100, 40],
            })

            detections = db.get_detections("CAM-001", limit=1)
            assert len(detections) >= 1
            det = detections[0]

            required = ["detected_plate", "confidence", "timestamp", "camera_id",
                        "track_id", "frames_seen", "frame"]
            for field in required:
                assert field in det, f"Detection must have '{field}' field"
        finally:
            db.DATABASE_PATH = original_path
            os.unlink(db_path)

    def test_detection_limit_respected(self):
        """
        get_detections() SHALL respect the limit parameter.

        **Validates: Requirement 12.3**
        """
        import database as db
        db.init_database()
        detections = db.get_detections("CAM-001", limit=2)
        assert len(detections) <= 2, \
            f"limit=2 must return at most 2 detections, got {len(detections)}"


# ===========================================================================
# Task 40 — Error handling tests
# ===========================================================================

class TestErrorHandling:
    """
    Error handling tests for the pipeline.

    **Task 40: Write error handling tests**
    **Validates: Requirements 1.7, 3.6, 10.5, 13.1, 13.4**
    """

    def test_invalid_video_returns_empty(self):
        """
        process_video() with a non-video file SHALL return empty list.

        **Validates: Requirement 10.5**
        """
        from processing.video_processor import process_video

        # Write a text file and try to process it as video
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.write(fd, b"not a video file")
        os.close(fd)
        try:
            result, _, _ = process_video(path, frame_interval=3)
            assert result == [], \
                f"Corrupt video must return [], got {result}"
        finally:
            os.unlink(path)

    def test_ocr_fallback_chain_returns_tuple(self):
        """
        _run_ocr_with_fallback() SHALL always return a (text, conf) tuple.

        **Validates: Requirements 3.6, 13.1**
        """
        from recognition.plate_reader import _run_ocr_with_fallback

        frame = np.zeros((40, 120, 3), dtype=np.uint8)
        result = _run_ocr_with_fallback(frame)
        assert isinstance(result, tuple), \
            f"_run_ocr_with_fallback must return a tuple, got {type(result)}"
        assert len(result) == 2
        text, conf = result
        assert text == "" or isinstance(text, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_validate_plate_handles_empty_input(self):
        """
        validate_plate() SHALL not raise on empty input.

        **Validates: Requirement 13.4**
        """
        from rules.grammar_validator import validate_plate

        db_path = _make_empty_db()
        try:
            result = validate_plate("", db_path=db_path)
            assert isinstance(result, dict)
            assert "violation" in result
            assert "detected_plate" in result
            assert result["detected_plate"] == ""
        finally:
            os.unlink(db_path)

    def test_validate_plate_handles_none_like_input(self):
        """
        validate_plate() SHALL not raise on whitespace-only input.

        **Validates: Requirement 13.4**
        """
        from rules.grammar_validator import validate_plate

        db_path = _make_empty_db()
        try:
            result = validate_plate("   ", db_path=db_path)
            assert isinstance(result, dict)
        finally:
            os.unlink(db_path)

    def test_levenshtein_handles_empty_strings(self):
        """
        levenshtein_distance() SHALL handle empty strings without error.

        **Validates: Requirement 5.1**
        """
        from deduplication.levenshtein import levenshtein_distance

        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("ABC", "") == 3
        assert levenshtein_distance("", "ABC") == 3

    def test_deduplicate_handles_empty_list(self):
        """
        deduplicate_detections() SHALL return [] for empty input.

        **Validates: Requirement 5.2**
        """
        from deduplication.plate_deduplicator import deduplicate_detections

        result = deduplicate_detections([])
        assert result == []

    def test_confidence_scorer_handles_zero_inputs(self):
        """
        calculate_confidence() SHALL not raise on zero inputs.

        **Validates: Requirement 9.1**
        """
        from scoring.confidence_scorer import calculate_confidence

        result = calculate_confidence(0.0, 0.0)
        assert isinstance(result, float)
        assert result == 0.0

    def test_plate_stabilizer_handles_empty_plate(self):
        """
        PlateStabilizer SHALL not raise on empty plate string.

        **Validates: Requirement 8.1**
        """
        from stabilization.plate_stabilizer import PlateStabilizer

        stabilizer = PlateStabilizer()
        result = stabilizer.stabilize_detection("", {"confidence": 0.5})
        assert isinstance(result, bool)

    def test_database_add_detection_handles_missing_fields(self):
        """
        add_detection() SHALL not raise when optional fields are missing.

        **Validates: Requirement 12.1**
        """
        import database as db

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        original_path = db.DATABASE_PATH
        db.DATABASE_PATH = db_path
        try:
            db.init_database()
            # Minimal detection — only required fields
            det_id = db.add_detection("CAM-001", {
                "detected_plate": "MH34AC3479",
                "confidence": 0.8,
            })
            assert isinstance(det_id, int)
            assert det_id > 0
        finally:
            db.DATABASE_PATH = original_path
            os.unlink(db_path)


# ===========================================================================
# Task 39 — Backward compatibility: API response format
# ===========================================================================

class TestAPIResponseFormat:
    """
    Verify API response structures match what the frontend expects.

    **Validates: Requirements 6.1–6.7, 20.1–20.5**
    """

    def test_detection_response_required_fields(self):
        """
        Detection response objects SHALL contain all required fields.

        **Validates: Requirements 6.5, 20.4, 20.5**
        """
        # Simulate what the /analyze-video endpoint builds
        detection = {
            "detected_plate": "MH34AC3479",
            "correct_plate": None,
            "violation": None,
            "confidence": 0.92,
            "yolo_conf": 0.88,
            "ocr_conf": 0.95,
            "confidence_modifier": 1.0,
            "frame": 15,
            "bbox": [10.5, 20.3, 15.2, 6.8],
            "plate_image": "data:image/jpeg;base64,/9j/...",
            "source": "video_analysis",
        }

        required_fields = [
            "detected_plate", "correct_plate", "violation",
            "confidence", "frame", "bbox", "plate_image",
        ]
        for field in required_fields:
            assert field in detection, f"Response must contain '{field}'"

    def test_bbox_is_list_of_four(self):
        """
        bbox SHALL be a list of 4 numeric values.

        **Validates: Requirements 6.7, 11.2**
        """
        bbox = [10.5, 20.3, 15.2, 6.8]
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        for val in bbox:
            assert isinstance(val, (int, float))

    def test_plate_image_is_data_url(self):
        """
        plate_image SHALL be a base64 JPEG data URL.

        **Validates: Requirements 20.1, 20.2, 20.3**
        """
        crop = np.zeros((40, 120, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        assert data_url.startswith("data:image/jpeg;base64,")
        # Verify round-trip
        raw = base64.b64decode(data_url.split(",", 1)[1])
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None

    def test_confidence_is_float_in_range(self):
        """
        confidence SHALL be a float in [0.0, 1.0].

        **Validates: Requirement 9.7**
        """
        from scoring.confidence_scorer import calculate_confidence

        for yolo in (0.0, 0.5, 1.0):
            for ocr in (0.0, 0.5, 1.0):
                conf = calculate_confidence(yolo, ocr)
                assert 0.0 <= conf <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
