# RoadVision — OCR Engines Setup Guide

To ensure high-accuracy license plate recognition, RoadVision uses a multi-engine fallback pipeline:
**CRNN → PaddleOCR → EasyOCR → Tesseract**

If you feel like OCR is "missing," it is usually because the external software or model weights are not installed.

## 1. Tesseract OCR (Windows Setup)
Tesseract is a powerful open-source engine. The code looks for it in specific Windows paths.

1.  **Download**: [Tesseract Installer for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
2.  **Install**: Run the `.exe` and install to the default path: `C:\Program Files\Tesseract-OCR`
3.  **Verify**: Open your terminal and type `tesseract --version`.
4.  **Backend Config**: The backend in `backend-python/recognition/crnn_recognizer.py` is already configured to look at:
    `C:\Program Files\Tesseract-OCR\tesseract.exe`

## 2. PaddleOCR & EasyOCR
These are Python-based deep learning engines.

1.  **Ensure Dependencies**: These require `torch` and `paddlepaddle`.
2.  **Install Command**:
    ```bash
    pip install paddlepaddle-tiny paddleocr easyocr
    ```
3.  **First Run**: The first time you run the backend, it will automatically download about 100-200MB of weights for these engines.

## 3. YOLOv8 Models (The "Things")
The object detection models are large binary files. 

1.  **Automatic Setup**: Run the included downloader script:
    ```bash
    python backend-python/download_models.py
    ```
2.  This will download `license_plate_detector.pt` and `yolov8n.pt` into the `models/` folder.

## 4. CRNN (Custom Model)
The `crnn.pth` file is a custom-trained model for specific fonts. If you don't have this file, the system is designed to **automatically skip it** and use the other three engines instead. It is not "broken" without it; it just uses the standard fallbacks.
