"""
Test video upload and processing.
"""

import requests

# Upload the test video
url = 'http://localhost:8000/analyze-video'
files = {'video': open('test_dashcam.mp4', 'rb')}

print('Uploading test video to backend...')
print('This may take 10-30 seconds...\n')

response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    detections = data.get('detections', [])
    
    print(f'✅ SUCCESS! Detected {len(detections)} plates\n')
    print('='*80)
    
    for i, det in enumerate(detections, 1):
        print(f'\nPlate {i}:')
        print(f'  Detected: {det["detected_plate"]}')
        print(f'  Correct:  {det["correct_plate"]}')
        violation = det["violation"] if det["violation"] else "None (LEGAL)"
        print(f'  Violation: {violation}')
        print(f'  Confidence: {det["confidence"]}')
        print(f'  Frame: {det["frame"]}')
    
    print('\n' + '='*80)
    
    violations = [d for d in detections if d['violation']]
    print(f'\nSummary:')
    print(f'  Total plates: {len(detections)}')
    print(f'  Legal plates: {len(detections) - len(violations)}')
    print(f'  Illegal plates: {len(violations)}')
    
    if len(detections) == 0:
        print('\n⚠️  No plates detected!')
        print('This could mean:')
        print('  - YOLO confidence threshold too high')
        print('  - Plates too small in video')
        print('  - Geometric filters too strict')
else:
    print(f'❌ Error: {response.status_code}')
    print(response.text)
