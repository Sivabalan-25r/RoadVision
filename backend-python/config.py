"""
RoadVision — Configuration Management
Centralized configuration for all model paths, thresholds, and system parameters.
"""

import os

# ============================================================================
# Detection Configuration (YOLOv26)
# ============================================================================

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'license_plate_detector.pt')

# YOLOv26 inference parameters
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Low threshold catches more plates; false positives filtered downstream
YOLO_IMAGE_SIZE = 320  # Inference resolution (320×320 for speed)
YOLO_USE_HALF_PRECISION = True  # Auto-detect CUDA for FP16 (set at runtime)

# Geometric filter thresholds
MIN_ASPECT_RATIO = 1.5   # Minimum width/height ratio
MAX_ASPECT_RATIO = 7.0   # Maximum width/height ratio
MIN_PLATE_WIDTH = 50     # Minimum crop width in pixels
MIN_PLATE_HEIGHT = 15    # Minimum crop height in pixels
MIN_PLATE_AREA = 750     # Minimum plate area in pixels
MAX_PLATE_AREA = 100000  # Maximum plate area in pixels
MAX_PLATE_AREA_RATIO = 0.15  # Maximum plate area as fraction of frame

# ============================================================================
# Tracking Configuration (BoT-SORT)
# ============================================================================

BOTSORT_MAX_AGE = 30  # Track expires after N frames without detection
BOTSORT_MIN_HITS = 1  # Assign track ID immediately
BOTSORT_IOU_THRESHOLD = 0.1  # Intersection-over-union threshold for matching

# ============================================================================
# OCR Configuration (PaddleOCR)
# ============================================================================

OCR_CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to accept OCR result
OCR_FALLBACK_CHAIN = ["paddleocr", "crnn", "easyocr", "tesseract"]

# OCR character corrections
OCR_CORRECTIONS = {
    '|': '1', 'l': '1', 'i': '1',
    'o': '0', 'q': '0',
    's': '5', 'z': '2', 'b': '8', 'g': '6',
    '(': 'C', ')': 'J', '[': 'L', ']': 'J',
}

# Text cleaning thresholds
MIN_CLEANED_LENGTH = 5
MAX_CLEANED_LENGTH = 12

# ============================================================================
# Validation Configuration
# ============================================================================

# Indian RTO plate format: AA NN AA NNNN
PLATE_FORMAT_PATTERN = r"^[A-Z]{2}\d{2}[A-Z0-9]{2,8}$"
MIN_PLATE_LENGTH = 6
MAX_PLATE_LENGTH = 12

# Known false positives
KNOWN_FALSE_POSITIVES = {
    'IND', 'INDIA', 'TEST', 'SAMPLE', 'DEMO',
    'GOVT', 'POLICE', 'TAXI', 'AUTO',
    'SHOP', 'STORE', 'MART', 'HOTEL', 'CAFE', 'OPEN', 'CLOSED',
    'PHONE', 'MOBILE', 'CALL', 'EXIT', 'ROAD', 'STREET', 'NAGAR',
}

# ============================================================================
# Stabilization Configuration
# ============================================================================

STABILIZATION_FRAMES = 2  # Require N frames for confirmation
TRACKER_EXPIRY_SECONDS = 30  # Expire tracker entries after N seconds

# ============================================================================
# Deduplication Configuration
# ============================================================================

LEVENSHTEIN_THRESHOLD = 2  # Maximum edit distance for duplicates
LENGTH_DIFF_THRESHOLD = 2  # Maximum length difference for duplicates

# ============================================================================
# Confidence Scoring Configuration
# ============================================================================

YOLO_WEIGHT = 0.4  # Weight for YOLO confidence in combined score
OCR_WEIGHT = 0.6  # Weight for OCR confidence in combined score
FORMAT_BOOST = 1.15  # Boost for plates matching Indian RTO format
STABILITY_BOOST_PER_FRAME = 0.05  # Boost per additional frame seen

# ============================================================================
# Video Processing Configuration
# ============================================================================

FRAME_INTERVAL = 3  # Process every Nth frame

# ============================================================================
# Preprocessing Configuration
# ============================================================================

TARGET_PLATE_WIDTH = 320  # Target width for plate preprocessing
TARGET_PLATE_HEIGHT = 120  # Target height for plate preprocessing

# ============================================================================
# Environment Variable Overrides
# ============================================================================

def load_env_overrides():
    """Load configuration overrides from environment variables."""
    import os
    
    global YOLO_CONFIDENCE_THRESHOLD, OCR_CONFIDENCE_THRESHOLD
    global STABILIZATION_FRAMES, LEVENSHTEIN_THRESHOLD, FRAME_INTERVAL
    
    if os.getenv('YOLO_CONF'):
        YOLO_CONFIDENCE_THRESHOLD = float(os.getenv('YOLO_CONF'))
    
    if os.getenv('OCR_CONF'):
        OCR_CONFIDENCE_THRESHOLD = float(os.getenv('OCR_CONF'))
    
    if os.getenv('STABILIZATION_FRAMES'):
        STABILIZATION_FRAMES = int(os.getenv('STABILIZATION_FRAMES'))
    
    if os.getenv('LEVENSHTEIN_THRESHOLD'):
        LEVENSHTEIN_THRESHOLD = int(os.getenv('LEVENSHTEIN_THRESHOLD'))
    
    if os.getenv('FRAME_INTERVAL'):
        FRAME_INTERVAL = int(os.getenv('FRAME_INTERVAL'))

# Load environment overrides on import
load_env_overrides()
