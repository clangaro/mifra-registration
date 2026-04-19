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

def compute_optical_flow(reference_frame, current_frame):
    """Compute dense optical flow between two frames.
    Returns a flow field where each pixel has (dx, dy) motion."""

    # convert to grayscale
    ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # compute optical flow using Farneback's method
    flow = cv2.calcOpticalFlowFarneback(
        ref_gray, curr_gray,
        None, # No initial flow
        0.5,  # pyramid scale
        3,    # levels
        15,   # window size
        3,    # iterations
        5,    # size of pixel neighborhood
        1.2,  # standard deviation of Gaussian
        0     # flags
    )
    return flow