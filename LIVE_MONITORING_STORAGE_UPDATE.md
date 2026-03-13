# Live Monitoring Storage Integration

## Overview
Updated the live monitoring system to automatically save all detected plates to localStorage, making them available in the Detections and History sections.

## Changes Implemented

### 1. Detection Storage Function
Added `saveDetectionToStorage(detection)` function that:
- Retrieves existing detections from localStorage
- Adds new detection with metadata:
  - `detected_plate`: The OCR-read plate number
  - `correct_plate`: Corrected/formatted plate
  - `violation`: Type of violation (if any)
  - `confidence`: Detection confidence score
  - `frame`: Timestamp as unique identifier
  - `bbox`: Bounding box coordinates
  - `timestamp`: ISO timestamp
  - `source`: 'live_monitoring' tag
- Saves updated array back to localStorage
- Prevents duplicate entries using `seenPlates` Set

### 2. Evidence Frame Capture
Added `captureAndSaveEvidence(detection)` function that:
- Captures current webcam frame as evidence
- Converts to JPEG data URL
- Saves using `saveEvidence()` from evidence-store.js
- Links evidence image to plate detection

### 3. Integration Points
Updated `processWebcamFrame()` to:
- Call `saveDetectionToStorage()` for each new plate
- Automatically capture and save evidence frames
- Log all saves to console for debugging

### 4. Page Load Enhancement
Added `loadExistingDetectionCount()` to:
- Check localStorage on page load
- Log existing detection count
- Prepare for future features (e.g., "Continue Session")

## Data Flow

```
Live Monitoring → Backend API → Detection Result
                                      ↓
                              saveDetectionToStorage()
                                      ↓
                              localStorage.setItem()
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
            Detections Page                      History Page
         (filters violations)                  (shows all)
```

## localStorage Structure

```javascript
{
  "roadvision_detections": [
    {
      "detected_plate": "TN10AB1234",
      "correct_plate": "TN 10 AB 1234",
      "violation": null,
      "confidence": 0.95,
      "frame": 1710345678901,
      "bbox": [35, 45, 18, 8],
      "timestamp": "2024-03-13T14:23:45.901Z",
      "source": "live_monitoring"
    },
    {
      "detected_plate": "MH12XY9876",
      "correct_plate": "MH 12 XY 9876",
      "violation": "Character Manipulation",
      "confidence": 0.88,
      "frame": 1710345682456,
      "bbox": [60, 30, 16, 7],
      "timestamp": "2024-03-13T14:23:49.456Z",
      "source": "live_monitoring"
    }
  ]
}
```

## Evidence Storage

Evidence frames are stored separately using the evidence-store.js component:

```javascript
{
  "roadvision_evidence": [
    {
      "plate": "TN10AB1234",
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    }
  ]
}
```

## How It Works

### Live Monitoring Session
1. User enables webcam
2. User clicks play to start detection
3. Every 1 second:
   - Frame captured from webcam
   - Sent to backend API
   - Backend returns detections
   - New plates saved to localStorage
   - Evidence frame captured and saved
4. User can navigate to Detections/History to see results

### Detections Page
- Reads from `localStorage.getItem('roadvision_detections')`
- Filters only violations: `detections.filter(d => d.violation)`
- Displays illegal plates with evidence images

### History Page
- Reads from `localStorage.getItem('roadvision_detections')`
- Shows all detections (legal + illegal)
- Displays full detection log with timestamps

## Features

### Automatic Saving
- ✅ No manual save required
- ✅ Real-time updates to localStorage
- ✅ Survives page refresh
- ✅ Persists across sessions

### Deduplication
- ✅ Uses `seenPlates` Set to track unique plates
- ✅ Only saves each plate once per session
- ✅ Prevents duplicate entries

### Evidence Capture
- ✅ Captures webcam frame at detection time
- ✅ Saves as JPEG data URL
- ✅ Available in evidence modal

### Source Tracking
- ✅ Tags detections with 'live_monitoring' source
- ✅ Distinguishes from video upload detections
- ✅ Enables future filtering/sorting

## Testing

### Test Live Monitoring Storage
1. Start backend: `cd backend-python && uvicorn main:app --reload`
2. Open monitoring page: `http://localhost:8000/static/monitoring.html`
3. Enable webcam and start detection
4. Wait for plates to be detected
5. Open browser console and check:
   ```javascript
   JSON.parse(localStorage.getItem('roadvision_detections'))
   ```
6. Navigate to Detections page - should see violations
7. Navigate to History page - should see all detections

### Test Evidence Frames
1. After detecting plates in live monitoring
2. Go to Detections page
3. Click "View Evidence" on any detection
4. Should see captured webcam frame

### Test Persistence
1. Detect some plates in live monitoring
2. Close browser tab
3. Reopen monitoring page
4. Check console - should log existing detection count
5. Go to History/Detections - data should still be there

## Clear Detection Data

To clear all stored detections (useful for testing):
```javascript
// In browser console
localStorage.removeItem('roadvision_detections');
localStorage.removeItem('roadvision_evidence');
location.reload();
```

## Future Enhancements

### Possible Additions
- Session management (start/stop/clear)
- Export detections to CSV/JSON
- Detection statistics dashboard
- Filter by date/time range
- Search by plate number
- Batch delete functionality

### Performance Considerations
- localStorage has ~5-10MB limit
- Consider implementing:
  - Automatic cleanup of old detections
  - Pagination for large datasets
  - IndexedDB for larger storage needs
  - Backend database sync

## Files Modified
- `monitoring.html` - Added storage and evidence capture functions

## Dependencies
- `components/evidence-store.js` - For evidence frame storage
- Browser localStorage API
- Backend API: `POST /api/process-frame`
