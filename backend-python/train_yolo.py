import os
import sys
from ultralytics import YOLO

def main():
    # 1. Dataset Path
    dataset_base = r"C:\Users\Sivab\.cache\kagglehub\datasets\sunrajbishnolia\license-plate-detection\versions\2\License Plate Detection"
    data_yaml_path = os.path.join(dataset_base, "data.yaml")
    
    # 2. Fix the data.yaml locally if needed
    # (Since we are using the absolute path in model.train, YOLO often handles it, 
    # but we'll ensure the YAML is clean)
    
    print(f"--- YOLO Training Initiated ---")
    print(f"Dataset: {dataset_base}")
    
    # 3. Initialize Model
    # We use YOLOv11 Nano (yolo11n.pt) as the base for 'yolov26'
    try:
        model = YOLO('yolo11n.pt') 
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # 4. Starting Training
    # Note: On CPU, this will be slow. Reduced imgsz and epochs for the mini-start.
    print("Starting fine-tuning (Mini-Run)...")
    
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=2,            # Mini-run for verification
            imgsz=320,           # Lower resolution for CPU speed
            batch=8,             # Small batch for memory safety
            device='cpu',        # Enforce CPU since no CUDA found
            workers=0,           # Stable for Windows CPU
            project='RoadVision_Models',
            name='yolov26_alpha'
        )
        print("Training mini-run completed successfully!")
        print(f"Results saved to: {results.save_dir}")
        
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
