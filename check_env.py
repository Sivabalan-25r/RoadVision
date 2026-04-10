import sys
import os

def check():
    print(f"Current Python Version: {sys.version}")
    print(f"Executable: {sys.executable}")
    
    major = sys.version_info.major
    minor = sys.version_info.minor
    
    if major == 3 and minor <= 13:
        print("\n✅ Python version is COMPATIBLE with PaddleOCR.")
        try:
            import paddle
            print("✅ PaddlePaddle is INSTALLED.")
            import paddleocr
            print("✅ PaddleOCR is INSTALLED.")
            print("\nResult: System is ready for ELITE accuracy mode.")
        except ImportError as e:
            print(f"\n❌ PaddleOCR is NOT installed yet.")
            print(f"Action: Run 'pip install paddlepaddle paddleocr'")
    else:
        print(f"\n❌ Python version {major}.{minor} is NOT compatible with PaddleOCR.")
        print("Action: Please install Python 3.13 and create a virtual environment.")

if __name__ == "__main__":
    check()
