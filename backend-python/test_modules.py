import sys
import os

# Ensure backend-python is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("--- Testing Kalman Tracker ---")
from stabilization.kalman_tracker import TrackerManager
tm = TrackerManager()
box = [100, 150, 200, 80]  # x, y, w, h
res1 = tm.update([box])
print(f"Update 1 result: {res1}")
for t_id, t in tm.trackers.items():
    print(f"Predict next bbox: {t.predict().tolist()}")

print("\n--- Testing Bayesian Arbitrator ---")
from recognition.bayesian_arbitrator import BayesianOCRArbitrator
arb = BayesianOCRArbitrator(threshold=0.65)
# Mock paddle to return low confidence (0.5 * 0.85 prior = 0.425 posterior < 0.65 threshold) -> Triggers EasyOCR
def mock_paddle(img):
    return "MH12AB1234", 0.5

def mock_easyocr(img):
    return "MH12AB1234", 0.9

res, conf = arb.arbitrate(None, None, mock_paddle, mock_easyocr)
print(f"Arbitrator Result: {res}")
print(f"Arbitrator Joint Confidence: {conf:.2f}")

print("\n--- Testing Positional Levenshtein ---")
from registration_db import lookup_vehicle
# Our DB has "MH12DE1433"
# Let's see if MN12DE1433 resolves successfully (1 state char error = cost 3, threshold=3)
res_valid = lookup_vehicle("MN12DE1433")
print(f"Testing 'MN12DE1433': Registered={res_valid.get('registered')}, Matched Plate={res_valid.get('plate_number')}")

# testing invalid
res_invalid = lookup_vehicle("XX99ZZ9999")
print(f"Testing 'XX99ZZ9999': Registered={res_invalid.get('registered')}")
