"""
Test validation rules on OCR output to see why violations aren't being detected.
"""

from rules.plate_rules import validate_plate, normalize_plate

# Test cases from OCR output
test_plates = [
    'IN82Y8388',      # Should be: IN 82 Y 8388 or similar
    'HR98AA7777',     # Should be: HR 98 AA 7777
    '1AAA00007',      # Starts with digit - should be corrected
    '09J7567',        # Starts with digit - should be corrected
    '1AAA0000',       # Starts with digit
    'I4440000',       # I should be 1
    '11A40000',       # Might be valid
    '891J4567',       # Starts with digit
    'WB65D18753',     # Example from user (should be valid)
]

print("=" * 80)
print("Testing Validation Rules")
print("=" * 80)

for plate in test_plates:
    print(f"\n📋 Testing: '{plate}'")
    print("-" * 80)
    
    # Normalize
    normalized = normalize_plate(plate)
    print(f"   Normalized: '{normalized}'")
    
    # Validate
    result = validate_plate(plate)
    print(f"   Detected:   '{result.detected_plate}'")
    print(f"   Correct:    '{result.correct_plate}'")
    print(f"   Violation:  {result.violation or 'None'}")
    print(f"   Modifier:   {result.confidence_modifier}")
    
    if result.violation:
        print(f"   ✅ VIOLATION DETECTED")
    else:
        print(f"   ⚠️  NO VIOLATION (plate appears valid)")

print("\n" + "=" * 80)
