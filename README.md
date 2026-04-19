# MIFRA Registration

A preprocessing tool for MIFRA microscopy videos. Converts rotating-sample 
footage into static-sample equivalent using optical flow registration, 
enabling MIFRA tensor analysis on grey matter tissue.

## What It Does

Standard MIFRA requires the microscope to mechanically rotate the tissue 
sample. This tool uses computer vision to computationally stabilise the 
tissue in post-processing, producing a video where the sample appears 
static and only the illumination appears to rotate. The output video 
plugs directly into the existing MIFRA pipeline at Stage 1.

## Installation

Requires the `mifra` conda environment (see main MIFRA repo for setup).

```bash
conda activate mifra
pip install opencv-python numpy
```

## Usage

```python
from register_frames import load_video, register_all_frames, save_video

frames = load_video("input_video.mp4")
registered = register_all_frames(frames, reference_index=0)
save_video(registered, "output_registered.mp4", fps=30)
```

Then feed `output_registered.mp4` into MIFRA as you would any raw video.

## Pipeline

1. **Load** — decode video into frame list
2. **Compute optical flow** — Farneback dense flow between reference and each frame
3. **Warp** — apply inverse flow to align each frame to reference
4. **Save** — encode registered frames back to mp4

## Testing

```bash
python tests/smoke_test.py
```

## Status

Prototype. Validated on 4K rotating-sample microscopy video (2215 frames). 
Grey matter validation pending.