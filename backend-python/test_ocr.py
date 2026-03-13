"""
Quick OCR test script to diagnose OCR accuracy issues.
Tests all available OCR engines on debug plate images.
"""

import cv2
import numpy as np
from pathlib import Path

# Test with a few debug images
debug_dir = Path("debug_plates")
test_images = [
    "frame0_plate0_raw.png",
    "frame0_plate0_processed.png",
    "frame10_plate0_raw.png",
    "frame10_plate0_processed.png",
]

print("=" * 60)
print("OCR Engine Comparison Test")
print("=" * 60)

for img_name in test_images:
    img_path = debug_dir / img_name
    if not img_path.exists():
        continue
    
    print(f"\n📸 Testing: {img_name}")
    print("-" * 60)
    
    img = cv2.imread(str(img_path))
    if img is None:
        print("  ❌ Failed to load image")
        continue
    
    print(f"  Image size: {img.shape}")
    
    # Test 1: EasyOCR
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(img, detail=1, paragraph=False)
        
        if results:
            texts = [text for (bbox, text, conf) in results if conf >= 0.3]
            combined = ''.join(texts).upper()
            avg_conf = sum(conf for (_, _, conf) in results) / len(results)
            print(f"  ✓ EasyOCR: '{combined}' (conf: {avg_conf:.2f})")
        else:
            print(f"  ✗ EasyOCR: No text detected")
    except Exception as e:
        print(f"  ✗ EasyOCR failed: {e}")
    
    # Test 2: Tesseract
    try:
        import pytesseract
        from PIL import Image
        
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Convert to PIL
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
        
        # Test with different configs
        config1 = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text1 = pytesseract.image_to_string(pil_img, config=config1).strip().upper()
        
        config2 = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text2 = pytesseract.image_to_string(pil_img, config=config2).strip().upper()
        
        print(f"  ✓ Tesseract PSM 7: '{text1}'")
        print(f"  ✓ Tesseract PSM 8: '{text2}'")
    except Exception as e:
        print(f"  ✗ Tesseract failed: {e}")
    
    # Test 3: PaddleOCR
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        results = ocr.ocr(img, cls=False)
        
        if results and results[0]:
            texts = []
            for line in results[0]:
                if line and len(line) >= 2:
                    if isinstance(line[1], tuple) and len(line[1]) >= 2:
                        text, conf = line[1][0], line[1][1]
                        if conf >= 0.5:
                            texts.append(str(text))
            
            combined = ''.join(texts).strip().upper()
            print(f"  ✓ PaddleOCR: '{combined}'")
        else:
            print(f"  ✗ PaddleOCR: No text detected")
    except Exception as e:
        print(f"  ✗ PaddleOCR failed: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
