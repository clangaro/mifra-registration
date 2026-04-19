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

