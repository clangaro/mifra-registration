import sys 
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                        

from register_frames import load_video  

def test_load_video():
    video_path = "/Users/carolinalangaro/Desktop/mifra_registration/data/MVI_6805.MP4"  # Path to a test video file
    frames = load_video(video_path)
    print(f"Loaded {len(frames)} frames from the video.")
    print(f"Frame shape: {frames[0].shape}")
    assert isinstance(frames, list), "Output should be a list of frames"
    assert len(frames) > 0, "There should be at least one frame loaded"
    print("load_video function works correctly.")
    assert all(isinstance(frame, (np.ndarray, type(None))) for frame in frames), "Each frame should be a numpy array or None"

if __name__ == "__main__":
        test_load_video()