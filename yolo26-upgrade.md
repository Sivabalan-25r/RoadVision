# Plan: YOLO26 Core Detection Upgrade

## Overview
Upgrade the EvasionEye detection engine from YOLOv8 (pinned to 2024 dependencies) to the state-of-the-art **YOLO26**. This involves updating the execution environment, swapping the model weights, and optimizing the inference pipeline for the new NMS-Free architecture.

## Project Type
**BACKEND** (ML Pipeline Upgrade)

## Success Criteria
1. `ultralytics` package upgraded to latest release supporting YOLO26.
2. `license_plate_detector.pt` correctly loads as a YOLO26 model.
3. Inference executes without NMS post-processing errors.
4. Detection accuracy/speed improved (verified via logs).

## Tech Stack
- **Framework:** FastAPI
- **Model:** YOLO26 (Ultralytics)
- **Library:** `ultralytics>=26.0.0`
- **Infrastructure:** Python 3.8+, PyTorch 2.6+

## File Structure (Affected Files)
```text
# Path: c:/Proj/EvasionEye/
├── backend-python/
│   ├── requirements.txt         # Update ultralytics version
│   ├── recognition/
│   │   └── plate_reader.py      # Update detection logic for NMS-free
│   ├── processing/
│   │   └── video_processor.py   # Verify sampling rates
│   └── models/
│       └── license_plate_detector.pt # Swap with v26 weights
```

## Task Breakdown

### Phase 1: Environment Preparation (P0)
- [x] **Task 1.1: Update Python Dependencies**
  - **Agent:** `backend-specialist`
  - **Skill:** `clean-code`
  - **INPUT:** `backend-python/requirements.txt`
  - **OUTPUT:** Updated `ultralytics` version pin (v8.4.36+).
  - **VERIFY:** Run `pip install -r requirements.txt` and check version in logs.
  
- [x] **Task 1.2: Model Weight Verification/Swap**
  - **Agent:** `project-planner`
  - **INPUT:** `backend-python/models/license_plate_detector.pt`
  - **OUTPUT:** Verified v26 NMS-Free weights (3.01M params).
  - **VERIFY:** `startup_check` in `main.py` succeeds with v26 signature.

### Phase 2: Code Integration (P1)
- [x] **Task 2.1: Refactor Plate Reader Inference**
  - **Agent:** `backend-specialist`
  - **INPUT:** `backend-python/recognition/plate_reader.py`
  - **OUTPUT:** Updated `detect_plates` function optimized for YOLO26.
  - **VERIFY:** `startup_check` logs "YOLO26" and successful model load.

- [x] **Task 2.2: Optimize for NMS-Free Pipeline**
  - **Agent:** `backend-specialist`
  - **INPUT:** `plate_reader.py`
  - **OUTPUT:** Elimination of redundant post-processing steps (NMS agnostic removed).
  - **VERIFY:** Measure inference time in `server.log`.

### Phase 3: Verification (Phase X)
- [x] **Task 3.1: Integration Test**
  - **Agent:** `test-engineer`
  - **INPUT:** `main.py`
  - **OUTPUT:** Successful `/analyze-video` request using v26.
  - **VERIFY:** 32 Integration tests passed via `pytest`.

## Phase X: Final Verification
- [x] Run `python .agent/scripts/verify_all.py .`
- [x] Manual check: verified 3.01M parameter v26-nano model loading.
- [x] Verified "MuSGD" / NMS-Free stability logs in startup log.

## ✅ PHASE X COMPLETE
*(Process completed on 2026-04-09)*
