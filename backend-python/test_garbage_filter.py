"""
Test garbage text filter to see which plates are being rejected.
"""

from recognition.plate_reader import is_garbage_text, clean_text

# Test cases from OCR output
test_plates = [
    'IN82Y8388',      
    'HR98AA7777',     
    '1AAA00007',      
    '09J7567',        
    '1AAA0000',       
    'I4440000',       
    '11A40000',       
    '891J4567',       
    'WB65D18753',     
]

print("=" * 80)
print("Testing Garbage Text Filter")
print("=" * 80)

for plate in test_plates:
    cleaned = clean_text(plate)
    is_garbage = is_garbage_text(cleaned)
    
    print(f"\n📋 '{plate}' → cleaned: '{cleaned}'")
    
    # Check individual conditions
    letter_count = sum(1 for c in cleaned if c.isalpha())
    digit_count = sum(1 for c in cleaned if c.isdigit())
    
    print(f"   Letters: {letter_count}, Digits: {digit_count}, Length: {len(cleaned)}")
    
    if is_garbage:
        print(f"   ❌ REJECTED as garbage")
    else:
        print(f"   ✅ PASSED garbage filter")

print("\n" + "=" * 80)
