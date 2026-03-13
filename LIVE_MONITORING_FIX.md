# Live Monitoring Fix - March 13, 2026

## Issue
When clicking the play button on the live monitoring page, mock detections (like "TN AB 1234") were showing on the static image instead of requiring webcam activation for real detection.

## Root Cause
The page was configured to show mock detections on the static image when detection was started without webcam. This was confusing because it looked like real detection but was just demo data.

## Fix Applied

### 1. Removed Mock Detection Mode
**Before**: Clicking play on static image would show mock detections every 3 seconds
**After**: Clicking play without webcam shows an alert asking user to enable webcam first

### 2. Clean Initial State
**Before**: Page loaded with mock detections already visible
**After**: Page loads with clean canvas (no detections shown)

### 3. Webcam-Only Detection
**Before**: Detection could run on static image (showing fake data)
**After**: Detection only works when webcam is active (showing real data)

## How to Use Now

### Step 1: Enable Webcam
1. Click the **camera icon** (videocam) button
2. Allow browser to access your webcam
3. You'll see your webcam feed replace the static image
4. Camera name changes to "Webcam Feed"

### Step 2: Start Detection
1. Click the **play button** (play_circle)
2. Live stats panel appears (Legal/Illegal/Total counts)
3. System captures frames every 1 second
4. Sends frames to backend for YOLO + OCR processing
5. Draws bounding boxes with plate numbers and status

### Step 3: View Results
- **Green boxes** = Legal plates
- **Red boxes** = Illegal plates (with violation type)
- **Blue boxes** = Detecting... (OCR in progress)
- **Stats panel** = Running count of legal/illegal plates

### Step 4: Stop Detection
1. Click the **pause button** (pause_circle)
2. Detection stops, stats panel hides
3. Canvas clears

### Step 5: Disable Webcam (Optional)
1. Click the **camera off icon** (videocam_off)
2. Webcam stops, returns to static image
3. Detection automatically stops

## What Changed in Code

### monitoring.html
```javascript
// OLD: Show mock detections on load
window.addEventListener('load', () => {
  resizeCanvas();
  fetchMockDetections(); // ❌ Shows fake data
});

// NEW: Clean canvas on load
window.addEventListener('load', () => {
  resizeCanvas();
  // Don't show mock detections - wait for user to start
});
```

```javascript
// OLD: Allow detection on static image
function startDetection() {
  if (isWebcamActive) {
    detectionInterval = setInterval(processWebcamFrame, 1000);
  } else {
    detectionInterval = setInterval(fetchMockDetections, 3000); // ❌ Fake data
  }
}

// NEW: Require webcam for detection
function startDetection() {
  currentDetections = [];
  drawDetections(currentDetections); // Clear canvas
  
  if (isWebcamActive) {
    detectionInterval = setInterval(processWebcamFrame, 1000);
  } else {
    alert('Please enable webcam to start real-time detection.');
    stopDetection();
  }
}
```

## Testing

### Test 1: Page Load
✅ Canvas should be empty (no bounding boxes)
✅ Static image visible
✅ No stats panel

### Test 2: Click Play Without Webcam
✅ Alert appears: "Please enable webcam to start real-time detection"
✅ Detection doesn't start
✅ Canvas remains empty

### Test 3: Enable Webcam
✅ Webcam feed replaces static image
✅ Camera name changes to "Webcam Feed"
✅ Camera icon changes to "videocam_off"

### Test 4: Click Play With Webcam
✅ Stats panel appears
✅ Play button changes to pause icon
✅ Frames sent to backend every 1 second
✅ Bounding boxes drawn on detected plates
✅ Plate numbers and status shown
✅ Stats update when new plates detected

### Test 5: Point at License Plate
✅ Blue box appears (Detecting...)
✅ After OCR completes, box turns green (legal) or red (illegal)
✅ Plate number displayed
✅ Violation type shown if illegal
✅ Stats increment

### Test 6: Stop Detection
✅ Pause button stops detection
✅ Stats panel hides
✅ Canvas clears
✅ Webcam still active

### Test 7: Disable Webcam
✅ Webcam stops
✅ Static image returns
✅ Detection stops automatically
✅ Canvas clears

## Expected Behavior

### Workflow
```
Page Load
  ↓
[Clean canvas, static image]
  ↓
Click Camera Icon
  ↓
[Webcam activates]
  ↓
Click Play Button
  ↓
[Detection starts, stats appear]
  ↓
Point at License Plate
  ↓
[Blue box → OCR → Green/Red box with plate number]
  ↓
Click Pause Button
  ↓
[Detection stops, stats hide]
  ↓
Click Camera Off Icon
  ↓
[Webcam stops, back to static image]
```

## Backend Integration

The page sends frames to `POST /api/process-frame` which:
1. Receives JPEG image
2. Runs YOLO detection
3. Runs OCR on detected plates
4. Validates plate format
5. Returns detections with bounding boxes (as percentages)

Response format:
```json
{
  "detections": [
    {
      "detected_plate": "HR98AA7777",
      "correct_plate": "HR 98 AA 7777",
      "violation": null,
      "confidence": 0.92,
      "bbox": [35.2, 45.8, 18.3, 8.1]
    }
  ]
}
```

## Notes

- Detection runs at 1 FPS (1 frame per second) to balance accuracy and performance
- Each frame takes ~300-500ms to process (YOLO + OCR)
- Plates are deduplicated (same plate won't increment stats twice)
- Stats persist until detection is stopped
- Webcam resolution: 1280×720 (720p)

## Files Modified

1. `monitoring.html`
   - Removed mock detection mode
   - Added webcam requirement check
   - Cleaned initial state

## Status

✅ Fixed - Live monitoring now only shows real detections from webcam
✅ No more mock/demo data confusion
✅ Clear user flow: Enable webcam → Start detection → See results

---

**Test it now:**
1. Go to http://localhost:3000/monitoring.html
2. Click camera icon to enable webcam
3. Click play button to start detection
4. Point webcam at a license plate
5. Watch the magic happen! 🎉
