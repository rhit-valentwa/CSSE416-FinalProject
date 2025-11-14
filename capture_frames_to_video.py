import cv2
import os
import glob

# Config
image_folder = 'captures/episode-1'
video_name = 'video.mp4'
fps = 30

# Get files and sort them (crucial for correct order)
images = glob.glob(os.path.join(image_folder, "episode-1_frame_*.png"))
images.sort()

if not images:
    print("No images found.")
    exit()

# Read first image to get dimensions
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1'
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

for image in images:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()
print(f"Video saved as {video_name}")