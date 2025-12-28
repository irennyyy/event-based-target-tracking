import cv2
import os
from natsort import natsorted

frame_folder = "C_event_frames"
output_video_path = "../../Old_video_compare/event_video.avi"
fps = 30# or 100

# Sort frames
frame_files = natsorted([
    os.path.join(frame_folder, f)
    for f in os.listdir(frame_folder)
    if f.endswith(".png")
])

# Check frame count
if not frame_files:
    raise RuntimeError("No PNG frames found in C_event_frames/ folder.")

# Load first frame to get size
first_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
height, width = first_frame.shape
frame_size = (width, height)

# Define Old_video_compare writer (XVID is broadly compatible)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

# Write all frames
for f in frame_files:
    gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    video_writer.write(bgr)

video_writer.release()
print(f" Video saved to: {output_video_path}")