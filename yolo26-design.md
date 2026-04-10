# Solution: YOLO26 Integration Design

## 1. Environment & Dependencies
The `backend-python/requirements.txt` currently pins `ultralytics` to an outdated version.
**Change:**
```diff
-ultralytics==8.2.0
+ultralytics>=26.0.0  # Official 2026 release with YOLO26 support
```
The system already has `torch>=2.1.0` and `torchvision>=0.16.0`, which meets the minimum requirements, but upgrading to latest stable is recommended for the **MuSGD** optimizer performance.

## 2. Model Loading Strategy (`plate_reader.py`)
Current `load_plate_model()` uses `ultralytics.YOLO`. This remains the same, but we will add logging to confirm the model version and architecture (e.g., NMS-Free).

## 3. Inference Logic Update
YOLO26's **NMS-Free** mode changes the post-inference requirements.
- **Current (v8):** Requires `conf` and `iou` thresholds for traditional N-M-S.
- **New (v26):** If using an end-to-end model, we can theoretically set `agnostic_nms=False` and potentially ignore `iou` as the model outputs one detection per object.
- **Preprocessing:** YOLO26 handles **Small-Target-Aware Label Assignment (STAL)** better, so we may be able to reduce the `YOLO_IMAGE_SIZE` from 640 to 480 or stick with 640 for maximum accuracy.

## 4. Stability Pipeline
The `PlateStabilizer` in `backend-python/stabilization/` will remain as a safety net to ensure OCR consistency, but the detection count from YOLO26 should be more reliable (fewer false positives/duplicates).

## 5. Deployment Script Updates
The `start.bat` and `start.sh` should ensure a fresh `pip install` to pick up the new library versions.
