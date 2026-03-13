"""
Create a simple test video with moving license plates for ANPR testing.
"""

import cv2
import numpy as np

# Video settings
width, height = 1280, 720
fps = 30
duration = 10  # seconds
total_frames = fps * duration

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_dashcam.mp4', fourcc, fps, (width, height))

# License plates
plates = [
    {'text': 'TN10AB1234', 'legal': True, 'y': 200, 'speed': 2},
    {'text': 'MH12CD5678', 'legal': False, 'y': 400, 'speed': 1.5},
    {'text': 'KA01EF9012', 'legal': False, 'y': 600, 'speed': 2.5}
]

print("Generating test video...")

for frame_num in range(total_frames):
    # Create road background (gray asphalt)
    frame = np.ones((height, width, 3), dtype=np.uint8) * 80
    
    # Add road lines
    cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 5)
    
    # Draw each vehicle with plate
    for i, plate in enumerate(plates):
        # Calculate position (moving right to left)
        x = width - int((frame_num * plate['speed']) % (width + 400))
        y = plate['y']
        
        # Draw vehicle rectangle (simplified)
        vehicle_color = (200, 200, 200) if i == 1 else (150, 150, 150)
        cv2.rectangle(frame, (x-100, y-80), (x+100, y+80), vehicle_color, -1)
        cv2.rectangle(frame, (x-100, y-80), (x+100, y+80), (0, 0, 0), 2)
        
        # Draw license plate
        plate_width = 200
        plate_height = 60
        plate_x = x - plate_width//2
        plate_y = y + 30
        
        # White plate background
        cv2.rectangle(frame, (plate_x, plate_y), 
                     (plate_x + plate_width, plate_y + plate_height), 
                     (255, 255, 255), -1)
        cv2.rectangle(frame, (plate_x, plate_y), 
                     (plate_x + plate_width, plate_y + plate_height), 
                     (0, 0, 0), 2)
        
        # Add plate text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(plate['text'], font, 0.8, 2)[0]
        text_x = plate_x + (plate_width - text_size[0]) // 2
        text_y = plate_y + (plate_height + text_size[1]) // 2
        
        if not plate['legal'] and i == 1:
            # Fancy font for illegal plate
            cv2.putText(frame, plate['text'], (text_x, text_y), 
                       cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.6, (0, 0, 0), 2)
        elif not plate['legal'] and i == 2:
            # Normal text but add tampering marks
            cv2.putText(frame, plate['text'], (text_x, text_y), 
                       font, 0.8, (0, 0, 0), 2)
            # Add scratch marks
            cv2.line(frame, (plate_x + 120, plate_y + 20), 
                    (plate_x + 160, plate_y + 40), (0, 0, 0), 4)
        else:
            # Legal plate - normal text
            cv2.putText(frame, plate['text'], (text_x, text_y), 
                       font, 0.8, (0, 0, 0), 2)
    
    # Add frame to video
    out.write(frame)
    
    if frame_num % 30 == 0:
        print(f"Progress: {frame_num}/{total_frames} frames")

out.release()
print(f"\n✅ Video created: test_dashcam.mp4")
print(f"Duration: {duration} seconds")
print(f"Resolution: {width}x{height}")
print(f"Plates: {len(plates)}")
print("\nYou can now upload this video to test your ANPR system!")
