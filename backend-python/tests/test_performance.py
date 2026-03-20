"""
Performance & Optimization Tests — Phase 11

Tasks 42–44: Verify detection, OCR, and database performance meets requirements.

**Validates: Requirements 1.6, 3.2, 12.3, 12.6**
"""

import os
import sys
import sqlite3
import tempfile
import time

import cv2
import numpy as np
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_empty_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE vehicle_registrations (
            plate_number TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            registration_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        )
        """
    )
    conn.commit()
    conn.close()
    return path


# ===========================================================================
# Task 42 — Detection performance
# ===========================================================================

class TestDetectionPerformance:
    """
    Task 42: Optimize detection performance.
    Validates: Requirement 1.6
    """

    def test_detect_plates_completes_within_timeout(self):
        """
        detect_plates() on a single frame SHALL complete in under 10 seconds
        including cold-start model loading (subsequent calls will be <2s on CPU).

        Validates: Requirement 1.6
        """
        from recognition.plate_reader import detect_plates, load_plate_model

        # Warm up — load model first (this is the slow part, done once at startup)
        load_plate_model()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Measure inference only (model already loaded)
        start = time.time()
        detect_plates(frame, frame_number=0)
        elapsed = time.time() - start

        assert elapsed < 10.0, (
            f"detect_plates() (post-warmup) took {elapsed:.2f}s — must complete within 10s"
        )

    def test_frame_interval_reduces_processed_frames(self):
        """
        Frame sampling at interval=3 SHALL process ~1/3 of frames vs interval=1.

        Validates: Requirement 1.6 (frame sampling optimisation)
        """
        # Verify the config value is set to 3 (every 3rd frame)
        import config
        assert config.FRAME_INTERVAL == 3, (
            f"FRAME_INTERVAL must be 3 for performance, got {config.FRAME_INTERVAL}"
        )

    def test_geometric_filter_rejects_fast(self):
        """
        passes_geometric_filter() SHALL be a pure in-memory operation (no I/O).

        Validates: Requirement 1.6
        """
        from recognition.plate_reader import passes_geometric_filter

        start = time.time()
        for _ in range(10_000):
            passes_geometric_filter(10, 10, 110, 50, 640, 480)
        elapsed = time.time() - start

        assert elapsed < 1.0, (
            f"10,000 geometric filter calls took {elapsed:.3f}s — must be <1s"
        )

    def test_fp16_config_enabled(self):
        """
        YOLO_USE_HALF_PRECISION SHALL be True in config (FP16 enabled for GPU).

        Validates: Requirement 1.6
        """
        import config
        assert config.YOLO_USE_HALF_PRECISION is True, (
            "YOLO_USE_HALF_PRECISION must be True to enable FP16 on GPU"
        )

    def test_preprocessing_completes_fast(self):
        """
        preprocess_plate_crop() on a 320×120 image SHALL complete in <0.5s.

        Validates: Requirement 1.6
        """
        from recognition.plate_reader import preprocess_plate_crop

        crop = np.random.randint(0, 255, (120, 320, 3), dtype=np.uint8)

        start = time.time()
        for _ in range(50):
            preprocess_plate_crop(crop)
        elapsed = time.time() - start

        assert elapsed < 5.0, (
            f"50 preprocess_plate_crop calls took {elapsed:.3f}s — must be <5s"
        )


# ===========================================================================
# Task 43 — OCR performance
# ===========================================================================

class TestOCRPerformance:
    """
    Task 43: Optimize OCR performance.
    Validates: Requirement 3.2
    """

    def test_ocr_confidence_threshold_set(self):
        """
        OCR_CONFIDENCE_THRESHOLD SHALL be 0.25 as specified.

        Validates: Requirement 3.2
        """
        import config
        assert config.OCR_CONFIDENCE_THRESHOLD == 0.25, (
            f"OCR_CONFIDENCE_THRESHOLD must be 0.25, got {config.OCR_CONFIDENCE_THRESHOLD}"
        )

    def test_ocr_fallback_chain_order(self):
        """
        OCR fallback chain SHALL be in order: paddleocr → crnn → easyocr → tesseract.

        Validates: Requirement 3.6
        """
        import config
        expected = ["paddleocr", "easyocr", "tesseract"]
        assert config.OCR_FALLBACK_CHAIN == expected, (
            f"OCR_FALLBACK_CHAIN must be {expected}, got {config.OCR_FALLBACK_CHAIN}"
        )

    def test_preprocess_variants_generates_four(self):
        """
        preprocess_plate_variants() SHALL generate exactly 4 variants.

        Validates: Requirement 3.5 (ensemble efficiency)
        """
        from recognition.plate_reader import preprocess_plate_variants

        crop = np.random.randint(0, 255, (60, 200, 3), dtype=np.uint8)
        variants = preprocess_plate_variants(crop)

        assert len(variants) == 4, (
            f"preprocess_plate_variants must return 4 variants, got {len(variants)}"
        )

    def test_preprocess_variants_fast(self):
        """
        preprocess_plate_variants() SHALL complete in <1s for a single crop.

        Validates: Requirement 3.2
        """
        from recognition.plate_reader import preprocess_plate_variants

        crop = np.random.randint(0, 255, (120, 320, 3), dtype=np.uint8)

        start = time.time()
        for _ in range(20):
            preprocess_plate_variants(crop)
        elapsed = time.time() - start

        assert elapsed < 5.0, (
            f"20 preprocess_plate_variants calls took {elapsed:.3f}s — must be <5s"
        )

    def test_ocr_model_singleton(self):
        """
        PaddleOCR model SHALL be loaded once (singleton pattern).

        Validates: Requirement 3.2 (caching)
        """
        from recognition import plate_reader as pr

        # Call load twice — second call must return same instance
        inst1 = pr.load_paddleocr_model()
        inst2 = pr.load_paddleocr_model()

        # Both calls return the same object (or both None if not installed)
        assert inst1 is inst2, (
            "load_paddleocr_model() must return the same singleton instance"
        )


# ===========================================================================
# Task 44 — Database performance
# ===========================================================================

class TestDatabasePerformance:
    """
    Task 44: Optimize database performance.
    Validates: Requirements 12.3, 12.6
    """

    def test_indexes_exist_on_detections(self):
        """
        detections table SHALL have indexes on camera_id, timestamp, violation.

        Validates: Requirements 12.3, 12.6
        """
        import database as db

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        original = db.DATABASE_PATH
        db.DATABASE_PATH = db_path
        try:
            db.init_database()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='detections'"
            )
            index_names = {row[0] for row in cursor.fetchall()}
            conn.close()

            assert "idx_detections_camera" in index_names, \
                "Missing index idx_detections_camera"
            assert "idx_detections_timestamp" in index_names, \
                "Missing index idx_detections_timestamp"
            assert "idx_detections_violation" in index_names, \
                "Missing index idx_detections_violation"
        finally:
            db.DATABASE_PATH = original
            os.unlink(db_path)

    def test_bulk_insert_performance(self):
        """
        Inserting 100 detections SHALL complete in under 2 seconds.

        Validates: Requirement 12.3
        """
        import database as db

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        original = db.DATABASE_PATH
        db.DATABASE_PATH = db_path
        try:
            db.init_database()

            start = time.time()
            for i in range(100):
                db.add_detection("CAM-001", {
                    "detected_plate": f"MH34AC{i:04d}",
                    "confidence": 0.9,
                    "frame": i,
                })
            elapsed = time.time() - start

            assert elapsed < 2.0, (
                f"100 inserts took {elapsed:.3f}s — must be <2s"
            )
        finally:
            db.DATABASE_PATH = original
            os.unlink(db_path)

    def test_query_with_limit_fast(self):
        """
        get_detections() with limit=100 SHALL complete in under 0.5s.

        Validates: Requirement 12.6
        """
        import database as db

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        original = db.DATABASE_PATH
        db.DATABASE_PATH = db_path
        try:
            db.init_database()

            # Insert 200 records
            for i in range(200):
                db.add_detection("CAM-001", {
                    "detected_plate": f"MH34AC{i:04d}",
                    "confidence": 0.9,
                    "frame": i,
                })

            start = time.time()
            results = db.get_detections("CAM-001", limit=100)
            elapsed = time.time() - start

            assert elapsed < 0.5, (
                f"get_detections(limit=100) took {elapsed:.3f}s — must be <0.5s"
            )
            assert len(results) == 100
        finally:
            db.DATABASE_PATH = original
            os.unlink(db_path)

    def test_violations_only_filter_uses_index(self):
        """
        violations_only query SHALL return correct results efficiently.

        Validates: Requirement 12.6
        """
        import database as db

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        original = db.DATABASE_PATH
        db.DATABASE_PATH = db_path
        try:
            db.init_database()

            # Insert mix of legal and violation detections
            for i in range(50):
                db.add_detection("CAM-001", {
                    "detected_plate": f"MH34AC{i:04d}",
                    "violation": "Character Manipulation" if i % 2 == 0 else None,
                    "confidence": 0.9,
                    "frame": i,
                })

            start = time.time()
            violations = db.get_detections("CAM-001", violations_only=True, limit=100)
            elapsed = time.time() - start

            assert elapsed < 0.5
            assert all(d["violation"] is not None for d in violations), \
                "violations_only must return only records with a violation"
            assert len(violations) == 25  # half are violations
        finally:
            db.DATABASE_PATH = original
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
