"""
Logging & Configuration Management
"""

import os
import yaml
import logging
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO"):
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# ---------------------------------------------------------------------------
# Configuration Management
# ---------------------------------------------------------------------------

class Config:
    """Manages application configuration with defaults and overrides."""
    
    DEFAULT_CONFIG = {
        "models": {
            "yolo": "models/best_v26.pt",
            "ocr_chain": ["paddle", "crnn", "easyocr", "tesseract"]
        },
        "thresholds": {
            "yolo_conf": 0.45,
            "ocr_conf": 0.40,
            "total_conf": 0.50
        },
        "stabilization": {
            "min_frames": 2,
            "expiry_sec": 30
        },
        "database": {
            "path": "roadvision.db",
            "registration_path": "registration_db.sqlite"
        }
    }

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.DEFAULT_CONFIG.copy()
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                overrides = yaml.safe_load(f)
                if overrides:
                    self._deep_update(self.config, overrides)
                    
    def _deep_update(self, base: Dict, overrides: Dict):
        for k, v in overrides.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a value using dot notation, e.g., 'models.yolo'."""
        keys = key_path.split(".")
        val = self.config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

# Global instance
config = Config()
