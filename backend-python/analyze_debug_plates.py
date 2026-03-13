"""
Analyze debug plate images to understand OCR performance.
Shows which plates are readable and which are false positives.
"""

import cv2
import numpy as np
from pathlib import Path
import easyocr

# Initialize EasyOCR
print("Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

debug_dir = Path("debug_plates")
raw_images = sorted(debug_dir.glob("*_raw.png"))

print(f"\n{'='*80}")
print(f"Analyzing {len(raw_images)} debug plate images")
print(f"{'='*80}\n")

readable_count = 0
too_small_count = 0
false_positive_count = 0

for img_path in raw_images[:20]:  # Analyze first 20
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    h, w = img.shape[:2]
    area = h * w
    
    print(f"📸 {img_path.name}")
    print(f"   Size: {w}x{h} (area: {area})")
    
    # Check if too small
    if w < 100 or h < 30 or area < 3000:
        print(f"   ❌ TOO SMALL - Would be filtered by new geometric constraints")
        too_small_count += 1
        print()
        continue
    
    # Try OCR
    try:
        # Preprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Upscale if needed
        if w < 200 or h < 60:
            scale = max(200 / w, 60 / h)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert to BGR
        bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # OCR
        results = reader.readtext(
            bgr,
            detail=1,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            batch_size=1
        )
        
        if results:
            texts = [text for (bbox, text, conf) in results if conf >= 0.2]
            combined = ''.join(texts).upper()
            avg_conf = sum(conf for (_, _, conf) in results) / len(results)
            
            # Check if looks like a plate
            has_letters = sum(1 for c in combined if c.isalpha())
            has_digits = sum(1 for c in combined if c.isdigit())
            
            if has_letters >= 1 and has_digits >= 1 and 6 <= len(combined) <= 12:
                print(f"   ✅ READABLE: '{combined}' (conf: {avg_conf:.2f})")
                readable_count += 1
            else:
                print(f"   ⚠️  SUSPICIOUS: '{combined}' (conf: {avg_conf:.2f}) - might be false positive")
                false_positive_count += 1
        else:
            print(f"   ❌ NO TEXT DETECTED - likely false positive")
            false_positive_count += 1
    
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        false_positive_count += 1
    
    print()

print(f"{'='*80}")
print(f"Summary:")
print(f"  ✅ Readable plates: {readable_count}")
print(f"  ❌ Too small (would be filtered): {too_small_count}")
print(f"  ⚠️  False positives / unreadable: {false_positive_count}")
print(f"{'='*80}")
print(f"\nRecommendation:")
if too_small_count > readable_count:
    print("  → Tighter geometric filters will help (already implemented)")
if false_positive_count > readable_count:
    print("  → YOLO is detecting many non-plates - consider higher confidence threshold")
if readable_count > 0:
    print(f"  → {readable_count} plates are readable - OCR should work on these")
