"""
Main script: run full registration pipeline on a video using streaming I/O.
"""

import cv2
import numpy as np
from register_frames import compute_optical_flow, warp_frame

# Configuration
INPUT_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/data/MVI_6805.MP4"
OUTPUT_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/data/MVI_6805_registered.mp4"
FPS = 30
DOWNSCALE_FACTOR = 0.5


def downscale_frame(frame, factor):
    """Downscale a single frame."""
    h, w = frame.shape[:2]
    new_size = (int(w * factor), int(h * factor))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)


def main():
    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input video: {total_frames} frames")
    
    # Read the first frame as reference
    ret, reference = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")
    
    if DOWNSCALE_FACTOR < 1.0:
        reference = downscale_frame(reference, DOWNSCALE_FACTOR)
    
    h, w = reference.shape[:2]
    print(f"Processing at resolution: {w}x{h}")
    
    # Open output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))
    
    # Write the reference frame first
    writer.write(reference)
    
    # Process remaining frames one at a time
    count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if DOWNSCALE_FACTOR < 1.0:
            frame = downscale_frame(frame, DOWNSCALE_FACTOR)
        
        # Register this frame
        flow = compute_optical_flow(reference, frame)
        warped = warp_frame(frame, flow)
        writer.write(warped)
        
        count += 1
        if count % 50 == 0:
            print(f"  Registered {count}/{total_frames} frames")
    
    cap.release()
    writer.release()
    print(f"Done! Saved {count} frames to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()