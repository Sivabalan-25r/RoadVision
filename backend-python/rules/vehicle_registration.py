"""
RoadVision — Vehicle Registration Lookup
Queries the SQLite vehicle_registrations table by plate number.

The vehicle_registrations table schema:
    plate_number      TEXT PRIMARY KEY
    owner             TEXT
    registration_date TEXT
    status            TEXT  (e.g. 'active', 'expired', 'suspended')
"""

import logging
import re
import sqlite3
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)

# Default path — can be overridden via VEHICLE_REGISTRATIONS_DB env var or
# by passing db_path explicitly to lookup_vehicle_registration().
_DEFAULT_DB_PATH = "registration_db.sqlite"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class VehicleRegistrationInfo(TypedDict):
    """Vehicle info returned by lookup_vehicle_registration()."""
    plate_number: str
    owner: str
    registration_date: str
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_plate(plate_text: str) -> str:
    """Uppercase and strip spaces/hyphens from a plate string."""
    return re.sub(r"[\s\-]+", "", plate_text.strip().upper())


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with row_factory set."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Create vehicle_registrations table if it does not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_registrations (
            plate_number      TEXT PRIMARY KEY,
            owner             TEXT NOT NULL,
            registration_date TEXT NOT NULL,
            status            TEXT NOT NULL DEFAULT 'active'
        )
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_vehicle_registration(
    plate_text: str,
    db_path: str = _DEFAULT_DB_PATH,
) -> Optional[VehicleRegistrationInfo]:
    """Look up a vehicle in the SQLite vehicle_registrations table.

    Normalises the plate string (uppercase, strips spaces/hyphens) before
    querying so that "MH 12 AB 1234" and "MH12AB1234" resolve to the same
    record.

    Args:
        plate_text: Raw plate string from OCR (may contain spaces/hyphens).
        db_path:    Path to the SQLite database file.  Defaults to
                    ``registration_db.sqlite`` in the current working directory.

    Returns:
        A :class:`VehicleRegistrationInfo` dict with keys
        ``plate_number``, ``owner``, ``registration_date``, ``status``
        when a matching record is found, or ``None`` when the plate is not
        registered.

    Examples:
        >>> result = lookup_vehicle_registration("MH12AB1234")
        >>> result is None or isinstance(result, dict)
        True
    """
    if not plate_text:
        logger.debug("lookup_vehicle_registration: empty plate text — returning None")
        return None

    plate = _normalize_plate(plate_text)

    try:
        conn = _get_connection(db_path)
        try:
            _ensure_table(conn)
            row = conn.execute(
                "SELECT plate_number, owner, registration_date, status "
                "FROM vehicle_registrations WHERE plate_number = ?",
                (plate,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.error(
            "lookup_vehicle_registration: database error for plate '%s': %s",
            plate, exc,
        )
        return None

    if row is None:
        logger.info(
            "lookup_vehicle_registration: plate '%s' NOT FOUND in vehicle_registrations",
            plate,
        )
        return None

    info = VehicleRegistrationInfo(
        plate_number=row["plate_number"],
        owner=row["owner"],
        registration_date=row["registration_date"],
        status=row["status"],
    )
    logger.info(
        "lookup_vehicle_registration: plate '%s' FOUND — owner='%s', "
        "registration_date='%s', status='%s'",
        plate, info["owner"], info["registration_date"], info["status"],
    )
    return info
