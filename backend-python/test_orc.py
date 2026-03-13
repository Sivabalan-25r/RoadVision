"""
Quick OCR test script to diagnose OCR accuracy issues.
Tests all available OCR engines on debug plate images.
"""

import cv2
import numpy as np
from pathlib import Path
import os

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
    
    print(f"\n[TESTING]: {img_name}")
    print("-" * 60)
    
    img = cv2.imread(str(img_path))
    if img is None:
        print("  [FAIL] Failed to load image")
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
            print(f"  [OK] EasyOCR: '{combined}' (conf: {avg_conf:.2f})")
        else:
            print(f"  [FAIL] EasyOCR: No text detected")
    except Exception as e:
        print(f"  [FAIL] EasyOCR failed: {e}")
    
    # Test 2: Tesseract
    try:
        import pytesseract
        from PIL import Image
        
        # Try local paths or assume it's in PATH
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Users\Sivab\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        ]
        tess_path = None
        for p in possible_paths:
            if os.path.exists(p):
                tess_path = p
                break
        
        if tess_path:
            pytesseract.pytesseract.tesseract_cmd = tess_path
            
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
        
        print(f"  [OK] Tesseract PSM 7: '{text1}'")
        print(f"  [OK] Tesseract PSM 8: '{text2}'")
    except Exception as e:
        print(f"  [FAIL] Tesseract failed: {e}")
    
    # Test 3: PaddleOCR
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        results = ocr.ocr(img, cls=False)
        
        if results and results[0]:
            texts: list[str] = []
            
            # Handle PaddleX dict format
            if isinstance(results[0], dict) and 'rec_texts' in results[0]:
                rec_texts = results[0].get('rec_texts', [])
                rec_scores = results[0].get('rec_scores', [])
                for text, conf in zip(rec_texts, rec_scores):
                    if conf >= 0.5:
                        texts.append(str(text))
            # Handle standard PaddleOCR format: [[[box], (text, conf)], ...]
            elif isinstance(results[0], list):
                for line in results[0]:
                    if line and len(line) >= 2:
                        if isinstance(line[1], tuple) and len(line[1]) >= 2:
                            text, conf = line[1][0], line[1][1]
                            if conf >= 0.5:
                                texts.append(str(text))
            
            combined = ''.join(texts).strip().upper()
            if combined:
                print(f"  [OK] PaddleOCR: '{combined}'")
            else:
                print(f"  [FAIL] PaddleOCR: No high-confidence text detected")
        else:
            print(f"  [FAIL] PaddleOCR: No text detected")
    except Exception as e:
        print(f"  [FAIL] PaddleOCR failed: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
