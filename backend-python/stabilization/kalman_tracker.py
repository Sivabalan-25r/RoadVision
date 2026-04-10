"""
EvasionEye — Kalman Filter Tracking
Implements NumPy-based Kalman filtering for bounding box stabilization
and missing frame prediction to reduce YOLO inference calls.
"""
import numpy as np

class KalmanPlateTracker:
    def __init__(self, track_id: int):
        self.track_id = track_id
        
        # State vector: [x, y, w, h, vx, vy, vw, vh]
        self.x = np.zeros((8, 1))
        
        # State transition matrix
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # Position updated by velocity
            
        # Measurement matrix (we only measure [x, y, w, h])
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0
            
        # Measurement Covariance (Measurement Noise)
        self.R = np.eye(4) * 0.1
        self.R[2:, 2:] *= 10.0  # w,h measurements are noisier than x,y
        
        # Process Covariance (Prediction Noise)
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 0.01  # Velocities don't change too rapidly
        
        # State Covariance
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 1000.0  # Initially, huge uncertainty in velocity
        
        self.is_initialized = False
        self.frames_since_update = 0

    def predict(self) -> np.ndarray:
        """Predict the next state."""
        # x = F * x
        self.x = np.dot(self.F, self.x)
        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.frames_since_update += 1
        return self.get_bbox()

    def update(self, bbox: np.ndarray):
        """Update state with a new measurement."""
        if not self.is_initialized:
            self.x[:4, 0] = bbox
            self.is_initialized = True
            self.frames_since_update = 0
            return
            
        # Measurement residual
        z = bbox.reshape(4, 1)
        y = z - np.dot(self.H, self.x)
        
        # System uncertainty
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        
        # Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update State
        self.x = self.x + np.dot(K, y)
        
        # Update Covariance
        I = np.eye(8)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        self.frames_since_update = 0

    def get_bbox(self) -> np.ndarray:
        """Return current bounding box format [x, y, w, h]."""
        return self.x[:4, 0].copy()
        
    def get_covariance(self) -> float:
        """Return a metric representing tracking uncertainty."""
        return float(np.trace(self.P))

class TrackerManager:
    def __init__(self, max_unseen=3, max_cov=20.0):
        self.trackers = {}
        self.next_id = 0
        self.max_unseen = max_unseen
        self.max_cov = max_cov

    def update(self, bboxes):
        """
        Takes YOLO detections, matches them to trackers via IoU, 
        and updates states. Returns current active states.
        """
        # If no trackers, init all
        if not self.trackers:
            for bbox in bboxes:
                t = KalmanPlateTracker(self.next_id)
                t.update(np.array(bbox))
                self.trackers[self.next_id] = t
                self.next_id += 1
            return [(t_id, t.get_bbox()) for t_id, t in self.trackers.items()]
            
        # Predict all
        predicted = {}
        for t_id, t in self.trackers.items():
            predicted[t_id] = t.predict()
            
        # Match using greedy IoU (simplified Hungarian)
        matched_bboxes = set()
        matched_trackers = set()
        
        for idx, bbox in enumerate(bboxes):
            best_iou = 0.3  # Threshold
            best_id = -1
            
            for t_id, p_bbox in predicted.items():
                if t_id in matched_trackers:
                    continue
                iou = self.compute_iou(bbox, p_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = t_id
                    
            if best_id != -1:
                self.trackers[best_id].update(np.array(bbox))
                matched_trackers.add(best_id)
                matched_bboxes.add(idx)
            else:
                # New track
                t = KalmanPlateTracker(self.next_id)
                t.update(np.array(bbox))
                self.trackers[self.next_id] = t
                matched_trackers.add(self.next_id)
                self.next_id += 1
                
        # Cleanup lost tracks
        lost_tracks = []
        for t_id, t in self.trackers.items():
            if t.frames_since_update > self.max_unseen or t.get_covariance() > self.max_cov:
                lost_tracks.append(t_id)
        for t_id in lost_tracks:
            del self.trackers[t_id]
            
        return [(t_id, t.get_bbox(), t.frames_since_update) for t_id, t in self.trackers.items()]

    def compute_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1+w1, x2+w2)
        yi2 = min(y1+h1, y2+h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        return inter_area / float(box1_area + box2_area - inter_area)
