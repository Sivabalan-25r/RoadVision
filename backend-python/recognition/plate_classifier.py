import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PlateColorClassifier:
    """
    Analyzes raw BGR crops of license plates using HSV color segmentation
    to classify them into Private, Commercial, EV, Rental, etc.
    """

    # Dominant background color ranges in HSV
    COLOR_RANGES = {
        'white':  ([0, 0, 168], [172, 111, 255]),
        'yellow': ([20, 100, 100], [35, 255, 255]),
        'green':  ([36, 50, 50], [89, 255, 255]),
        'red':    ([0, 120, 70], [10, 255, 255]),
        'blue':   ([90, 50, 50], [130, 255, 255]),
        'black':  ([0, 0, 0], [180, 255, 50]),
    }

    # Text color ranges in HSV
    TEXT_COLOR_RANGES = {
        'white_text': ([0, 0, 200], [180, 30, 255]),
        'yellow_text': ([20, 100, 100], [35, 255, 255]),
        'black_text': ([0, 0, 0], [180, 255, 50]),
    }

    def __init__(self):
        pass

    def classify(self, raw_bgr_crop: np.ndarray) -> dict:
        """
        Detects background and text color to determine plate type.
        
        Returns:
            dict: {
                "plate_type": str,
                "color": str,
                "text_color": str,
                "confidence": float,
                "is_hsrp": bool
            }
        """
        if raw_bgr_crop is None or raw_bgr_crop.size == 0:
            return self._default_result()

        hsv = cv2.cvtColor(raw_bgr_crop, cv2.COLOR_BGR2HSV)
        
        # 1. Detect Background Color
        bg_color, bg_conf = self._detect_dominant_color(hsv, self.COLOR_RANGES)
        
        # 2. Detect Text Color (focus on center-ish area to avoid borders)
        h, w = hsv.shape[:2]
        roi = hsv[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        text_color, text_conf = self._detect_dominant_color(roi, self.TEXT_COLOR_RANGES)
        
        # 3. Determine Plate Type based on color combination
        plate_type = self._map_to_type(bg_color, text_color)
        
        # 4. HSRP Detection (Hologram + Indicator)
        hsrp_data = self.detect_hsrp(raw_bgr_crop)
        is_hsrp = hsrp_data["hsrp"]

        return {
            "plate_type": plate_type,
            "color": bg_color,
            "text_color": text_color.replace("_text", ""),
            "confidence": round(float(bg_conf * 0.7 + text_conf * 0.3), 2),
            "is_hsrp": is_hsrp,
            "hsrp_confidence": hsrp_data["hsrp_confidence"]
        }

    def _detect_dominant_color(self, hsv_img, color_dict) -> tuple:
        best_color = "unknown"
        max_percent = 0
        
        total_pixels = hsv_img.shape[0] * hsv_img.shape[1]
        
        for color_name, (lower, upper) in color_dict.items():
            mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
            count = cv2.countNonZero(mask)
            percent = count / total_pixels
            
            if percent > max_percent:
                max_percent = percent
                best_color = color_name
        
        return best_color, max_percent

    def _map_to_type(self, bg: str, text: str) -> str:
        # Table from requirements
        if bg == 'white' and text == 'black_text':
            return "Private Vehicle"
        if bg == 'yellow' and text == 'black_text':
            return "Commercial/Transport"
        if bg == 'black' and text == 'yellow_text':
            return "Self-Drive Rental"
        if bg == 'green' and text == 'white_text':
            return "Electric (Private)"
        if bg == 'green' and text == 'yellow_text':
            return "Electric (Commercial)"
        if bg == 'red' and text == 'white_text':
            return "Temporary Registration"
        if bg == 'blue' and text == 'white_text':
            return "Diplomatic Vehicle"
        
        # Fallbacks
        if bg == 'white': return "Private Vehicle"
        if bg == 'yellow': return "Commercial/Transport"
        
        return "Unknown"

    def detect_hsrp(self, plate_crop) -> dict:
        """
        Comprehensive HSRP verification:
        1. Detect Ashoka Chakra hologram using HoughCircles in top-left ROI.
        2. Detect permanent identification number (not implemented yet, requires OCR).
        3. Check blue IND strip.
        """
        h, w = plate_crop.shape[:2]
        if h < 20 or w < 60:
            return {"hsrp": False, "hsrp_confidence": 0.0}

        # --- 1. Ashoka Chakra Hologram Detection ---
        # Ashoka Chakra is in top-left ~15% of plate width, top 40% of height
        roi_h = int(h * 0.4)
        roi_w = int(w * 0.15)
        roi = plate_crop[0:roi_h, 0:roi_w]
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Blur to reduce noise for better circle detection
        gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        
        circles = cv2.HoughCircles(
            gray_roi, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=10, param1=50, param2=30,
            minRadius=3, maxRadius=int(roi_h * 0.8)
        )
        
        hologram_detected = circles is not None and len(circles[0]) >= 1
        
        # --- 2. Blue IND Strip Check ---
        blue_ind = self._check_hsrp_indicator(plate_crop)
        
        # Combined logic: HSRP plates MUST have both (or high confidence on one)
        # Chromium hologram is the strongest indicator of HSRP
        hsrp_detected = hologram_detected or blue_ind
        
        return {
            "hsrp": hsrp_detected,
            "hsrp_confidence": 0.85 if hologram_detected else (0.5 if blue_ind else 0.20)
        }

    def _check_hsrp_indicator(self, bgr_crop) -> bool:
        """
        Check for the blue 'IND' strip on the left side of the plate.
        """
        h, w = bgr_crop.shape[:2]
        left_strip = bgr_crop[:, :int(w*0.15)]
        hsv_strip = cv2.cvtColor(left_strip, cv2.COLOR_BGR2HSV)
        
        # Blue Range for IND strip
        blue_lower = np.array([100, 150, 50])
        blue_upper = np.array([140, 255, 255])
        
        mask = cv2.inRange(hsv_strip, blue_lower, blue_upper)
        blue_pixels = cv2.countNonZero(mask)
        total_strip_pixels = left_strip.shape[0] * left_strip.shape[1]
        
        if total_strip_pixels == 0: return False
        blue_percent = blue_pixels / total_strip_pixels
        
        return blue_percent > 0.08

    def _default_result(self):
        return {
            "plate_type": "Unknown",
            "color": "unknown",
            "text_color": "unknown",
            "confidence": 0.0,
            "is_hsrp": False
        }
