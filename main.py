"""
Main script: run full registration pipeline on a video using feature-based
registration with temporal smoothing to reduce trembling artefacts.
"""

import cv2
import numpy as np
from register_frames import (
    compute_rigid_transform,
    warp_frame_affine,
    apply_circular_crop,
    smooth_transforms,
)

# Configuration
INPUT_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/data/MVI_6805_cut.mp4"
OUTPUT_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/output-videos/MVI_6805_registered.mp4"
FPS = 30
DOWNSCALE_FACTOR = 0.5

# Temporal smoothing to reduce trembling.
# Higher window = smoother but slower response to real rotation changes.
SMOOTHING_WINDOW = 5

# Circular crop to remove border artefacts from rotation.
APPLY_CIRCULAR_CROP = True
CROP_RADIUS_FACTOR = 1.0


def downscale_frame(frame, factor):
    """Downscale a single frame."""
    h, w = frame.shape[:2]
    new_size = (int(w * factor), int(h * factor))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)


def main():
    # ---- PASS 1: compute all transformation matrices ----
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input video: {total_frames} frames")

    ret, reference = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    if DOWNSCALE_FACTOR < 1.0:
        reference = downscale_frame(reference, DOWNSCALE_FACTOR)

    h, w = reference.shape[:2]
    print(f"Processing at resolution: {w}x{h}")
    print("Pass 1: computing transformation matrices...")

    # Identity matrix for the reference frame
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    matrices = [identity]
    failed = 0
    count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if DOWNSCALE_FACTOR < 1.0:
            frame = downscale_frame(frame, DOWNSCALE_FACTOR)

        try:
            matrix = compute_rigid_transform(reference, frame)
            matrices.append(matrix)
        except RuntimeError as e:
            print(f"  Warning: registration failed on frame {count}: {e}")
            matrices.append(None)
            failed += 1

        count += 1
        if count % 50 == 0:
            print(f"  Computed {count}/{total_frames} transforms")

    cap.release()
    print(f"Pass 1 complete. {count} transforms computed, {failed} failed.")

    # ---- Smooth the sequence of transforms ----
    print(f"Smoothing transforms with window size {SMOOTHING_WINDOW}...")
    matrices = smooth_transforms(matrices, window_size=SMOOTHING_WINDOW)

    # ---- PASS 2: warp and write frames using smoothed transforms ----
    print("Pass 2: warping and writing frames...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if DOWNSCALE_FACTOR < 1.0:
            frame = downscale_frame(frame, DOWNSCALE_FACTOR)

        matrix = matrices[count]
        warped = warp_frame_affine(frame, matrix)

        if APPLY_CIRCULAR_CROP:
            warped = apply_circular_crop(warped, radius_factor=CROP_RADIUS_FACTOR)

        writer.write(warped)
        count += 1
        if count % 50 == 0:
            print(f"  Wrote {count}/{total_frames} frames")

    cap.release()
    writer.release()
    print(f"Done! Saved {count} frames to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()