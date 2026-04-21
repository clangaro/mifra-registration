"""
Main script: run full registration pipeline on a video using feature-based
registration with temporal smoothing.

Registration is performed at low resolution for speed, then the resulting
transforms are scaled up and applied at a higher output resolution.
"""

import cv2
import numpy as np
import os
from register_frames import (
    compute_rigid_transform,
    warp_frame_affine,
    apply_circular_crop,
    smooth_transforms,
    scale_transform_matrix,
    save_trial_metadata,
)

# Configuration
INPUT_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/data/MVI_6805_cut.mp4"
OUTPUT_DIR = "/Users/carolinalangaro/Desktop/mifra_registration/output-videos"
OUTPUT_BASENAME = "test"
FPS = 30

# Resolution control — two separate factors
REGISTRATION_SCALE = 0.5   # Used for feature detection and matching (speed)
OUTPUT_SCALE = 1.0          # Used for the final warped video (quality)

# Temporal smoothing to reduce trembling
SMOOTHING_WINDOW = 21

# Circular crop to remove border artefacts from rotation
APPLY_CIRCULAR_CROP = True
CROP_RADIUS_FACTOR = 1.0

# Notes for this trial
TRIAL_NOTES = "Registration at 0.5x, output at 1.0x (full res). Smoothing window 15."


def downscale_frame(frame, factor):
    """Downscale a single frame."""
    h, w = frame.shape[:2]
    new_size = (int(w * factor), int(h * factor))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)


def get_next_versioned_path(directory, basename, extension=".mp4"):
    """Find the next available versioned filename in a directory."""
    os.makedirs(directory, exist_ok=True)
    existing = os.listdir(directory)

    max_trial = 0
    prefix = f"{basename}-"
    for filename in existing:
        if filename.startswith(prefix) and filename.endswith(extension):
            try:
                trial_str = filename[len(prefix):-len(extension)]
                trial_num = int(trial_str)
                max_trial = max(max_trial, trial_num)
            except ValueError:
                continue

    next_trial = max_trial + 1
    return os.path.join(directory, f"{basename}-{next_trial}{extension}")


def main():
    output_video = get_next_versioned_path(OUTPUT_DIR, OUTPUT_BASENAME)
    print(f"Output file: {output_video}")

    # ---- PASS 1: compute transforms at low resolution ----
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input video: {total_frames} frames")

    # Read and downscale reference for registration
    ret, reference_full = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    reference_reg = downscale_frame(reference_full, REGISTRATION_SCALE)
    h_reg, w_reg = reference_reg.shape[:2]
    print(f"Registration resolution: {w_reg}x{h_reg} (scale {REGISTRATION_SCALE})")

    # Identity matrix for the reference frame
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    matrices = [identity]
    failed = 0
    count = 1

    print("Pass 1: computing transforms at low resolution...")
    while True:
        ret, frame_full = cap.read()
        if not ret:
            break

        frame_reg = downscale_frame(frame_full, REGISTRATION_SCALE)

        try:
            matrix = compute_rigid_transform(reference_reg, frame_reg)
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

    # ---- Scale transforms up to output resolution ----
    scale_up = OUTPUT_SCALE / REGISTRATION_SCALE
    print(f"Scaling transforms by {scale_up}x for output resolution...")
    matrices = [scale_transform_matrix(m, scale_up) for m in matrices]

    # ---- PASS 2: warp and write at output resolution ----
    print(f"Pass 2: warping at output scale {OUTPUT_SCALE}...")
    cap = cv2.VideoCapture(INPUT_VIDEO)

    # Determine output dimensions from first frame at output scale
    ret, first_frame = cap.read()
    if OUTPUT_SCALE < 1.0:
        first_frame_out = downscale_frame(first_frame, OUTPUT_SCALE)
    else:
        first_frame_out = first_frame
    h_out, w_out = first_frame_out.shape[:2]
    print(f"Output resolution: {w_out}x{h_out}")

    # Rewind and set up writer
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, FPS, (w_out, h_out))

    count = 0
    while True:
        ret, frame_full = cap.read()
        if not ret:
            break

        if OUTPUT_SCALE < 1.0:
            frame_out = downscale_frame(frame_full, OUTPUT_SCALE)
        else:
            frame_out = frame_full

        matrix = matrices[count]
        warped = warp_frame_affine(frame_out, matrix)

        if APPLY_CIRCULAR_CROP:
            warped = apply_circular_crop(warped, radius_factor=CROP_RADIUS_FACTOR)

        writer.write(warped)
        count += 1
        if count % 50 == 0:
            print(f"  Wrote {count}/{total_frames} frames")

    cap.release()
    writer.release()
    print(f"Done! Saved {count} frames to {output_video}")

    # Save metadata
    config = {
        "input_video": INPUT_VIDEO,
        "fps": FPS,
        "registration_scale": REGISTRATION_SCALE,
        "output_scale": OUTPUT_SCALE,
        "smoothing_window": SMOOTHING_WINDOW,
        "apply_circular_crop": APPLY_CIRCULAR_CROP,
        "crop_radius_factor": CROP_RADIUS_FACTOR,
        "total_frames": count,
        "registration_resolution": f"{w_reg}x{h_reg}",
        "output_resolution": f"{w_out}x{h_out}",
    }
    save_trial_metadata(output_video, config, notes=TRIAL_NOTES)


if __name__ == "__main__":
    main()