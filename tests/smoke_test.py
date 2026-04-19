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

def test_optical_flow():
    from register_frames import compute_optical_flow

    video_path = "/Users/carolinalangaro/Desktop/mifra_registration/data/MVI_6805.MP4"  # Path to a test video file
    frames = load_video(video_path)

    # compute flow between first two frames
    flow = compute_optical_flow(frames[0], frames[1])

    print(f"Optical flow shape: {flow.shape}")
    print(f"Flow type: {flow.dtype}")
    print(f"Max motion (x): {np.max(flow[..., 0].max()):.2f} pixels")
    print(f"Max motion (y): {np.max(flow[..., 1].max()):.2f} pixels")

    assert flow.shape[:2] == frames[0].shape[:2], "Flow should have same height and width as input frames"
    assert flow.shape[2] == 2, "Flow should have two channels (dx, dy)"
    print("compute_optical_flow function works correctly.")

def test_warp_frame():
    from register_frames import compute_optical_flow, warp_frame
    
    video_path = "/Users/carolinalangaro/Desktop/mifra_registration/data/MVI_6805.MP4"
    frames = load_video(video_path)
    
    reference = frames[0]
    current = frames[1]
    
    # Compute flow from reference to current
    flow = compute_optical_flow(reference, current)
    
    # Warp current frame back to align with reference
    warped = warp_frame(current, flow)
    
    print(f"Original shape: {current.shape}")
    print(f"Warped shape: {warped.shape}")
    print(f"Warped dtype: {warped.dtype}")
    
    assert warped.shape == current.shape, "Warped frame should match original shape"
    assert warped.dtype == current.dtype, "Warped frame should have same dtype"
    print("warp_frame works correctly.")

if __name__ == "__main__":
    test_load_video()
    test_optical_flow()
    test_warp_frame()
