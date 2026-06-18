import cv2
import os
import numpy as np

base_path = "./generated_dataset/OmniBotics_V1_Reaching_2000_Episodes"
episodes = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

# Pick 4 spread-out episodes
sample_indices = [0, 500, 1000, 1500]
frames = []

for idx in sample_indices:
    video_path = os.path.join(base_path, episodes[idx], "rgb.mp4")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read() # Get the very first frame
    if ret:
        # Add a label to the frame
        cv2.putText(frame, f"Ep {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frames.append(frame)
    cap.release()

if len(frames) == 4:
    # Create a 2x2 grid
    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    grid = np.vstack((top_row, bottom_row))
    
    cv2.imwrite("diversity_check.jpg", grid)
    print("\n✅ Comparison image 'diversity_check.jpg' created!")
    print("Open this image to see the differences in objects, textures, and lighting.")
else:
    print("❌ Could not find enough episodes to compare.")
