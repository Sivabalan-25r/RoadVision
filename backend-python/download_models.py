import os
import urllib.request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("model_downloader")

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Model URLs from known public sources
MODEL_URLS = {
    'license_plate_detector.pt': 'https://huggingface.co/keremberke/yolov8-license-plate/resolve/main/best.pt',
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt'
}

def download_file(url, target_path):
    """Download a file from a URL to a target local path."""
    try:
        logger.info(f"Downloading {url} to {target_path}...")
        urllib.request.urlretrieve(url, target_path)
        logger.info(f"Successfully downloaded {target_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download from {url}: {e}")
        return False

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.info(f"Created models directory: {MODELS_DIR}")

    # Check and download models
    for filename, url in MODEL_URLS.items():
        target_path = os.path.join(MODELS_DIR, filename)
        
        if os.path.exists(target_path):
            size = os.path.getsize(target_path) / (1024 * 1024)
            logger.info(f"✓ {filename} already exists ({size:.2f} MB)")
        else:
            logger.info(f"✗ {filename} is missing.")
            download_file(url, target_path)

    # Note on CRNN.pth
    crnn_path = os.path.join(MODELS_DIR, 'crnn.pth')
    if not os.path.exists(crnn_path):
        logger.warning("-" * 60)
        logger.warning("NOTE: 'crnn.pth' (Custom CRNN weights) is not available via direct URL.")
        logger.warning("The system will automatically fall back to PaddleOCR / EasyOCR / Tesseract.")
        logger.warning(f"If you have custom weights, place them at: {crnn_path}")
        logger.warning("-" * 60)

    logger.info("Done!")

if __name__ == "__main__":
    main()
