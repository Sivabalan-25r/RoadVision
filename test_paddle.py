from paddleocr import PaddleOCR
import numpy as np

# Reverting to user's original test script for version 2.x
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
dummy = np.zeros((100, 300, 3), dtype=np.uint8)
result = ocr.ocr(dummy, cls=True)
print("PaddleOCR working:", result)
