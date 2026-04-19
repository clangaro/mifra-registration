"""
Main script: run full registration pipeline on a video using streaming I/O
and feature-based registration. Outputs both the registered video and a
validity mask video indicating which pixels are valid in each frame.
"""

import cv2
import numpy as np
from register_frames import compute_rigid_transform, warp_frame_affine, compute_validity_mask

# Configuration
INPUT_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/data/MVI_6805_cut.mp4"
OUTPUT_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/output-videos/MVI_6805_registered.mp4"
MASK_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/output-videos/MVI_6805_validity_mask.mp4"
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

    # Open output video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))
    mask_writer = cv2.VideoWriter(MASK_VIDEO, fourcc, FPS, (w, h), isColor=False)

    # Write the reference frame (fully valid by definition)
    writer.write(reference)
    ref_mask = np.full((h, w), 255, dtype=np.uint8)
    mask_writer.write(ref_mask)

    # Track cumulative validity: pixels that are valid in ALL frames so far
    cumulative_valid = ref_mask.copy()

    # Process remaining frames one at a time
    count = 1
    failed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if DOWNSCALE_FACTOR < 1.0:
            frame = downscale_frame(frame, DOWNSCALE_FACTOR)

        try:
            matrix = compute_rigid_transform(reference, frame)
            warped = warp_frame_affine(frame, matrix)
            mask = compute_validity_mask(warped)
            writer.write(warped)
            mask_writer.write(mask)
            cumulative_valid = cv2.bitwise_and(cumulative_valid, mask)
        except RuntimeError as e:
            print(f"  Warning: registration failed on frame {count}: {e}")
            writer.write(frame)
            full_mask = np.full((h, w), 255, dtype=np.uint8)
            mask_writer.write(full_mask)
            failed += 1

        count += 1
        if count % 50 == 0:
            print(f"  Registered {count}/{total_frames} frames")

    cap.release()
    writer.release()
    mask_writer.release()

    # Report coverage statistics
    valid_pixels = np.sum(cumulative_valid > 0)
    total_pixels = h * w
    coverage = (valid_pixels / total_pixels) * 100
    print(f"Done! Saved {count} frames to {OUTPUT_VIDEO}")
    print(f"Validity mask video saved to {MASK_VIDEO}")
    print(f"Pixels valid across ALL frames: {valid_pixels:,} / {total_pixels:,} ({coverage:.1f}%)")
    if failed > 0:
        print(f"Note: {failed} frames failed registration.")


if __name__ == "__main__":
    main()