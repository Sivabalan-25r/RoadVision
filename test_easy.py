import easyocr
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)
dummy = np.zeros((100, 300, 3), dtype=np.uint8)
result = reader.readtext(dummy)
print("EasyOCR working:", result)
