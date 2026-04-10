import os
import sys

# Add backend-python to path
sys.path.append(os.path.join(os.getcwd(), 'backend-python'))

try:
    from ultralytics import YOLO
    import torch
    
    model_path = os.path.join('backend-python', 'models', 'license_plate_detector.pt')
    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
        print(f"Model version: {model.task}")
        # Inspect model task and architecture
        print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()) / 1e6:.2f}M")
        
        # Check if it has 'nms' in its architecture (end-to-end models don't)
        has_nms = any('nms' in str(m).lower() for m in model.model.modules())
        print(f"Contains NMS modules: {has_nms}")
        
    else:
        print(f"File not found: {model_path}")

except Exception as e:
    print(f"Error: {e}")
