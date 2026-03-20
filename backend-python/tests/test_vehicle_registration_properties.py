"""
Property-Based Tests for Vehicle Registration Lookup

**Property 45: Vehicle Registration Lookup**
**Validates: Requirements 18.1, 18.2, 18.3**

Requirement 18.1: The system shall query the vehicle_registrations table by
plate number and return vehicle info when found.

Requirement 18.2: The system shall return None when the plate is not found in
the vehicle_registrations table.

Requirement 18.3: The returned vehicle info shall include owner,
registration_date, and status fields.

## Properties Tested

- **Property 45a**: lookup_vehicle_registration() returns None for plates not
  in the database.

- **Property 45b**: When a plate is found, the result contains the required
  fields: owner, registration_date, status.

- **Property 45c**: When a plate is found, all required fields are non-empty
  strings.

- **Property 45d**: Plate normalisation — lookups with spaces/hyphens resolve
  to the same record as the bare plate.

- **Property 45e**: Empty or whitespace-only plate text returns None without
  raising an exception.
"""

import os
import re
import sqlite3
import sys
import tempfile

import pytest
from hypothesis import given, settings, strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.vehicle_registration import lookup_vehicle_registration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(records: list[tuple]) -> str:
    """Create a temporary SQLite DB with vehicle_registrations rows.

    Args:
        records: list of (plate_number, owner, registration_date, status)

    Returns:
        Path to the temporary database file.
    """
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
    conn.executemany(
        "INSERT INTO vehicle_registrations VALUES (?, ?, ?, ?)", records
    )
    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid Indian plate characters (uppercase letters + digits)
_PLATE_CHARS = st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

@st.composite
def plate_string(draw, min_len=6, max_len=12):
    """Generate a random plate-like string (uppercase alphanumeric, 6–12 chars)."""
    length = draw(st.integers(min_value=min_len, max_value=max_len))
    chars = draw(st.lists(_PLATE_CHARS, min_size=length, max_size=length))
    return "".join(chars)


@st.composite
def plate_with_spaces(draw):
    """Generate a plate string with random spaces/hyphens inserted."""
    bare = draw(plate_string())
    # Insert a space or hyphen at a random position
    pos = draw(st.integers(min_value=1, max_value=len(bare) - 1))
    sep = draw(st.sampled_from([" ", "-"]))
    return bare[:pos] + sep + bare[pos:]


@st.composite
def owner_name(draw):
    """Generate a plausible owner name."""
    first = draw(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll")), min_size=2, max_size=10))
    last = draw(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll")), min_size=2, max_size=10))
    return f"{first} {last}"


@st.composite
def registration_date(draw):
    """Generate a date string in YYYY-MM-DD format."""
    year = draw(st.integers(min_value=2000, max_value=2024))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))
    return f"{year:04d}-{month:02d}-{day:02d}"


@st.composite
def reg_status(draw):
    """Generate a registration status string."""
    return draw(st.sampled_from(["active", "expired", "suspended"]))


# ---------------------------------------------------------------------------
# Property 45: Vehicle Registration Lookup
# ---------------------------------------------------------------------------

class TestVehicleRegistrationLookupProperties:
    """
    Property-based tests for lookup_vehicle_registration().

    **Property 45: Vehicle Registration Lookup**
    **Validates: Requirements 18.1, 18.2, 18.3**
    """

    # ------------------------------------------------------------------
    # Property 45a — plates not in DB return None
    # ------------------------------------------------------------------

    @given(plate=plate_string())
    @settings(max_examples=100, deadline=5000)
    def test_property_45a_unknown_plate_returns_none(self, plate):
        """
        **Property 45a: lookup_vehicle_registration() returns None for unknown plates**

        For any plate that was never inserted into the database,
        the function must return None.

        **Validates: Requirements 18.2**
        """
        # Empty DB — no records
        db_path = _make_db([])
        try:
            result = lookup_vehicle_registration(plate, db_path=db_path)
            assert result is None, (
                f"Plate '{plate}' not in DB — expected None, got {result}"
            )
        finally:
            os.unlink(db_path)

    # ------------------------------------------------------------------
    # Property 45b — found plates return a dict with required fields
    # ------------------------------------------------------------------

    @given(
        plate=plate_string(),
        owner=owner_name(),
        reg_date=registration_date(),
        status=reg_status(),
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_45b_found_plate_has_required_fields(
        self, plate, owner, reg_date, status
    ):
        """
        **Property 45b: Found plates return a dict with required fields**

        When a plate is present in the database, lookup_vehicle_registration()
        must return a dict containing 'owner', 'registration_date', and 'status'.

        **Validates: Requirements 18.1, 18.3**
        """
        db_path = _make_db([(plate, owner, reg_date, status)])
        try:
            result = lookup_vehicle_registration(plate, db_path=db_path)
            assert result is not None, (
                f"Plate '{plate}' is in DB — expected a dict, got None"
            )
            assert "owner" in result, "Result must contain 'owner'"
            assert "registration_date" in result, "Result must contain 'registration_date'"
            assert "status" in result, "Result must contain 'status'"
        finally:
            os.unlink(db_path)

    # ------------------------------------------------------------------
    # Property 45c — required fields are non-empty strings
    # ------------------------------------------------------------------

    @given(
        plate=plate_string(),
        owner=owner_name(),
        reg_date=registration_date(),
        status=reg_status(),
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_45c_found_plate_fields_are_nonempty_strings(
        self, plate, owner, reg_date, status
    ):
        """
        **Property 45c: Required fields are non-empty strings**

        When a plate is found, owner, registration_date, and status must all
        be non-empty strings (not None, not empty).

        **Validates: Requirements 18.3**
        """
        db_path = _make_db([(plate, owner, reg_date, status)])
        try:
            result = lookup_vehicle_registration(plate, db_path=db_path)
            assert result is not None

            assert isinstance(result["owner"], str) and result["owner"], (
                f"'owner' must be a non-empty string, got {result['owner']!r}"
            )
            assert isinstance(result["registration_date"], str) and result["registration_date"], (
                f"'registration_date' must be a non-empty string, got {result['registration_date']!r}"
            )
            assert isinstance(result["status"], str) and result["status"], (
                f"'status' must be a non-empty string, got {result['status']!r}"
            )
        finally:
            os.unlink(db_path)

    # ------------------------------------------------------------------
    # Property 45d — plate normalisation (spaces/hyphens stripped)
    # ------------------------------------------------------------------

    @given(
        plate=plate_string(),
        owner=owner_name(),
        reg_date=registration_date(),
        status=reg_status(),
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_45d_plate_normalisation_spaces(
        self, plate, owner, reg_date, status
    ):
        """
        **Property 45d: Plate normalisation — spaces are stripped before lookup**

        A plate stored as 'MH12AB1234' must be found when queried as
        'MH 12 AB 1234' (with spaces).

        **Validates: Requirements 18.1**
        """
        # Store the bare plate; query with a space inserted
        if len(plate) < 2:
            return  # skip trivially short plates
        spaced = plate[:2] + " " + plate[2:]
        db_path = _make_db([(plate, owner, reg_date, status)])
        try:
            result = lookup_vehicle_registration(spaced, db_path=db_path)
            assert result is not None, (
                f"Querying '{spaced}' should find plate '{plate}', got None"
            )
        finally:
            os.unlink(db_path)

    @given(
        plate=plate_string(),
        owner=owner_name(),
        reg_date=registration_date(),
        status=reg_status(),
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_45d_plate_normalisation_lowercase(
        self, plate, owner, reg_date, status
    ):
        """
        **Property 45d: Plate normalisation — lowercase input resolves correctly**

        A plate stored as 'MH12AB1234' must be found when queried in lowercase.

        **Validates: Requirements 18.1**
        """
        db_path = _make_db([(plate, owner, reg_date, status)])
        try:
            result = lookup_vehicle_registration(plate.lower(), db_path=db_path)
            assert result is not None, (
                f"Querying '{plate.lower()}' should find plate '{plate}', got None"
            )
        finally:
            os.unlink(db_path)

    # ------------------------------------------------------------------
    # Property 45e — empty / whitespace input returns None without error
    # ------------------------------------------------------------------

    @given(empty=st.one_of(st.just(""), st.just("   "), st.just("\t"), st.just("-")))
    @settings(max_examples=20, deadline=2000)
    def test_property_45e_empty_plate_returns_none(self, empty):
        """
        **Property 45e: Empty or whitespace-only plate text returns None**

        Passing an empty string or whitespace must not raise an exception and
        must return None.

        **Validates: Requirements 18.2**
        """
        db_path = _make_db([])
        try:
            result = lookup_vehicle_registration(empty, db_path=db_path)
            assert result is None, (
                f"Empty/whitespace plate '{empty!r}' must return None, got {result}"
            )
        finally:
            os.unlink(db_path)

    # ------------------------------------------------------------------
    # Spot-checks: known inputs
    # ------------------------------------------------------------------

    def test_spot_check_known_plate_found(self):
        """
        **Property 45: Known plate is found with correct fields**

        **Validates: Requirements 18.1, 18.3**
        """
        db_path = _make_db([("MH12AB1234", "Sanjay Patil", "2020-06-15", "active")])
        try:
            result = lookup_vehicle_registration("MH12AB1234", db_path=db_path)
            assert result is not None
            assert result["owner"] == "Sanjay Patil"
            assert result["registration_date"] == "2020-06-15"
            assert result["status"] == "active"
        finally:
            os.unlink(db_path)

    def test_spot_check_unknown_plate_returns_none(self):
        """
        **Property 45: Unknown plate returns None**

        **Validates: Requirements 18.2**
        """
        db_path = _make_db([("MH12AB1234", "Sanjay Patil", "2020-06-15", "active")])
        try:
            result = lookup_vehicle_registration("XX99ZZ9999", db_path=db_path)
            assert result is None
        finally:
            os.unlink(db_path)

    def test_spot_check_spaced_plate_found(self):
        """
        **Property 45: Plate with spaces resolves to stored record**

        **Validates: Requirements 18.1**
        """
        db_path = _make_db([("MH12AB1234", "Sanjay Patil", "2020-06-15", "active")])
        try:
            result = lookup_vehicle_registration("MH 12 AB 1234", db_path=db_path)
            assert result is not None
            assert result["owner"] == "Sanjay Patil"
        finally:
            os.unlink(db_path)

    def test_spot_check_expired_status_returned(self):
        """
        **Property 45: Expired status is returned correctly**

        **Validates: Requirements 18.3**
        """
        db_path = _make_db([("DL01AA4321", "Amit Verma", "2018-03-10", "expired")])
        try:
            result = lookup_vehicle_registration("DL01AA4321", db_path=db_path)
            assert result is not None
            assert result["status"] == "expired"
        finally:
            os.unlink(db_path)

    def test_spot_check_empty_string_returns_none(self):
        """
        **Property 45: Empty string returns None**

        **Validates: Requirements 18.2**
        """
        db_path = _make_db([])
        try:
            result = lookup_vehicle_registration("", db_path=db_path)
            assert result is None
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
