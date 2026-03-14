"""
RoadVision Database Module
SQLite database for storing camera accounts and detections
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

DATABASE_PATH = "roadvision.db"


def get_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Cameras table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            location TEXT,
            resolution TEXT DEFAULT '1080p',
            latitude TEXT,
            longitude TEXT,
            address TEXT,
            accuracy TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Detections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            detected_plate TEXT NOT NULL,
            correct_plate TEXT,
            violation TEXT,
            confidence REAL,
            frame INTEGER,
            bbox TEXT,
            plate_image TEXT,
            source TEXT DEFAULT 'live_monitoring',
            vehicle_registered BOOLEAN DEFAULT 0,
            vehicle_owner TEXT,
            vehicle_type TEXT,
            vehicle_state TEXT,
            vehicle_model TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
        )
    """)
    
    # Create indexes for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_camera 
        ON detections(camera_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_timestamp 
        ON detections(timestamp DESC)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_violation 
        ON detections(violation)
    """)
    
    conn.commit()
    conn.close()
    
    logger.info("Database initialized successfully")
    
    # Insert default cameras if they don't exist
    insert_default_cameras()


def insert_default_cameras():
    """Insert default camera accounts"""
    default_cameras = [
        {
            "camera_id": "CAM-001",
            "name": "Camera 1",
            "location": "Main Expressway North",
        },
        {
            "camera_id": "CAM-002",
            "name": "Camera 2",
            "location": "City Center Junction",
        },
        {
            "camera_id": "CAM-003",
            "name": "Camera 3",
            "location": "Highway Toll Plaza",
        },
        {
            "camera_id": "CAM-004",
            "name": "Camera 4",
            "location": "Airport Road Entry",
        },
    ]
    
    conn = get_connection()
    cursor = conn.cursor()
    
    for camera in default_cameras:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO cameras (camera_id, name, location)
                VALUES (?, ?, ?)
            """, (camera["camera_id"], camera["name"], camera["location"]))
        except sqlite3.IntegrityError:
            pass  # Camera already exists
    
    conn.commit()
    conn.close()


def get_all_cameras() -> List[Dict]:
    """Get all cameras"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM cameras ORDER BY camera_id")
    cameras = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return cameras


def get_camera(camera_id: str) -> Optional[Dict]:
    """Get camera by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
    row = cursor.fetchone()
    
    conn.close()
    return dict(row) if row else None


def update_camera(camera_id: str, data: Dict):
    """Update camera information"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE cameras 
        SET name = ?, location = ?, resolution = ?, 
            latitude = ?, longitude = ?, address = ?, accuracy = ?
        WHERE camera_id = ?
    """, (
        data.get("name"),
        data.get("location"),
        data.get("resolution"),
        data.get("latitude"),
        data.get("longitude"),
        data.get("address"),
        data.get("accuracy"),
        camera_id
    ))
    
    conn.commit()
    conn.close()


def add_detection(camera_id: str, detection: Dict):
    """Add a new detection"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Extract vehicle info if present
    vehicle_info = detection.get("vehicle_info", {})
    
    cursor.execute("""
        INSERT INTO detections 
        (camera_id, detected_plate, correct_plate, violation, confidence, 
         frame, bbox, plate_image, source, vehicle_registered, vehicle_owner, 
         vehicle_type, vehicle_state, vehicle_model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        camera_id,
        detection.get("detected_plate"),
        detection.get("correct_plate"),
        detection.get("violation"),
        detection.get("confidence"),
        detection.get("frame"),
        json.dumps(detection.get("bbox")) if detection.get("bbox") else None,
        detection.get("plate_image"),
        detection.get("source", "live_monitoring"),
        vehicle_info.get("registered", False),
        vehicle_info.get("owner_name"),
        vehicle_info.get("vehicle_type"),
        vehicle_info.get("state"),
        vehicle_info.get("model"),
    ))
    
    conn.commit()
    detection_id = cursor.lastrowid
    conn.close()
    
    return detection_id


def get_detections(camera_id: str, limit: int = 100, violations_only: bool = False) -> List[Dict]:
    """Get detections for a camera"""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT * FROM detections 
        WHERE camera_id = ?
    """
    
    if violations_only:
        query += " AND violation IS NOT NULL"
    
    query += " ORDER BY timestamp DESC LIMIT ?"
    
    cursor.execute(query, (camera_id, limit))
    detections = []
    
    for row in cursor.fetchall():
        det = dict(row)
        # Parse bbox JSON
        if det.get("bbox"):
            det["bbox"] = json.loads(det["bbox"])
        
        # Reconstruct vehicle_info if present
        if det.get("vehicle_registered") is not None:
            det["vehicle_info"] = {
                "registered": bool(det.get("vehicle_registered")),
                "owner_name": det.get("vehicle_owner"),
                "vehicle_type": det.get("vehicle_type"),
                "state": det.get("vehicle_state"),
                "model": det.get("vehicle_model"),
            }
        
        detections.append(det)
    
    conn.close()
    return detections


def get_detection_stats(camera_id: str) -> Dict:
    """Get detection statistics for a camera"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total detections
    cursor.execute("""
        SELECT COUNT(*) as total FROM detections WHERE camera_id = ?
    """, (camera_id,))
    total = cursor.fetchone()["total"]
    
    # Violations
    cursor.execute("""
        SELECT COUNT(*) as violations FROM detections 
        WHERE camera_id = ? AND violation IS NOT NULL
    """, (camera_id,))
    violations = cursor.fetchone()["violations"]
    
    # Average confidence
    cursor.execute("""
        SELECT AVG(confidence) as avg_confidence FROM detections 
        WHERE camera_id = ?
    """, (camera_id,))
    avg_conf = cursor.fetchone()["avg_confidence"]
    
    conn.close()
    
    return {
        "total_detections": total,
        "violations": violations,
        "legal_plates": total - violations,
        "avg_confidence": round(avg_conf * 100, 1) if avg_conf else 0
    }


def clear_all_detections(camera_id: str):
    """Clear all detections for a camera"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM detections WHERE camera_id = ?", (camera_id,))
    
    conn.commit()
    conn.close()


def delete_detection(detection_id: int):
    """Delete a specific detection"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
    
    conn.commit()
    conn.close()
