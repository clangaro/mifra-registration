"""
Main script: run full registration pipeline on a video using streaming I/O
and feature-based registration.
"""

import cv2
import numpy as np
from register_frames import compute_rigid_transform, warp_frame_affine, apply_circular_crop

# Configuration
INPUT_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/data/MVI_6805_cut.mp4"
OUTPUT_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/output-videos/MVI_6805_registered.mp4"
FPS = 30
DOWNSCALE_FACTOR = 0.5 # 1.0 = full res, <1.0 = downscale (e.g. 0.5 for half res). Adjust to speed up processing at the cost of some accuracy.

# Circular crop to remove border artefacts from rotation.
# Set APPLY_CIRCULAR_CROP = False to keep full frame (useful if peripheral tissue is needed).
# Adjust CROP_RADIUS_FACTOR to tune how much is kept.
APPLY_CIRCULAR_CROP = True
CROP_RADIUS_FACTOR = 1.0


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

    # Write the reference frame first (with crop if enabled)
    if APPLY_CIRCULAR_CROP:
        reference_out = apply_circular_crop(reference, radius_factor=CROP_RADIUS_FACTOR)
    else:
        reference_out = reference
    writer.write(reference_out)

    # Process remaining frames one at a time
    count = 1
    failed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if DOWNSCALE_FACTOR < 1.0:
            frame = downscale_frame(frame, DOWNSCALE_FACTOR)

        # Register this frame using feature-based matching
        try:
            matrix = compute_rigid_transform(reference, frame)
            warped = warp_frame_affine(frame, matrix)
            if APPLY_CIRCULAR_CROP:
                warped = apply_circular_crop(warped, radius_factor=CROP_RADIUS_FACTOR)
            writer.write(warped)
        except RuntimeError as e:
            print(f"  Warning: registration failed on frame {count}: {e}")
            if APPLY_CIRCULAR_CROP:
                frame = apply_circular_crop(frame, radius_factor=CROP_RADIUS_FACTOR)
            writer.write(frame)
            failed += 1

        count += 1
        if count % 50 == 0:
            print(f"  Registered {count}/{total_frames} frames")

    cap.release()
    writer.release()
    print(f"Done! Saved {count} frames to {OUTPUT_VIDEO}")
    if failed > 0:
        print(f"Note: {failed} frames failed registration and were left unaligned.")


if __name__ == "__main__":
    main()