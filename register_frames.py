import cv2
import numpy as np

def load_video(video_path):
    """Load all frames from video file"""

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

if __name__ == "__main__":
    video_path = "input_video.mp4"  # Path to your input video
    frames = load_video(video_path)

    print(f"Loaded {len(frames)} frames from {video_path}")
    print(f"Frame shape: {frames[0].shape}")  # Print shape of the first frame

    