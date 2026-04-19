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

def warp_frame(frame, flow):
    """
    Warp the input frame according to the given flow field.
    """

    h, w = frame.shape[:2]

    # create a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # add flow to pixel coordinates
    x_warped = (x + flow[..., 0]).astype(np.float32)
    y_warped = (y + flow[..., 1]).astype(np.float32)

    # warp the frame using remap
    warped = cv2.remap(
        frame,
        x_warped,
        y_warped,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return warped

def register_all_frames(frames, reference_index=0, verbose=True):
    """
    Register all frames to a reference frame using optical flow.
    
    Parameters:
    -----------
    frames : list of np.ndarray
        List of video frames
    reference_index : int
        Index of the frame to use as the static reference
    verbose : bool
        Print progress updates
    
    Returns:
    --------
    registered_frames : list of np.ndarray
        Frames aligned to the reference
    """
    reference = frames[reference_index]
    registered = []
    
    for i, frame in enumerate(frames):
        if i == reference_index:
            # Reference frame doesn't need warping
            registered.append(frame)
        else:
            flow = compute_optical_flow(reference, frame)
            warped = warp_frame(frame, flow)
            registered.append(warped)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Registered {i + 1}/{len(frames)} frames")
    
    if verbose:
        print(f"✓ Registered all {len(frames)} frames")
    
    return registered

def compute_rigid_transform(reference_frame, current_frame, max_features=5000):
    """
    Compute rigid transform (rotation + translation) between two frames
    using ORB feature detection and matching.
    
    Returns a 2x3 affine transformation matrix.
    """
    # Convert to grayscale
    ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(cur_gray, None)
    
    if des1 is None or des2 is None:
        raise RuntimeError("No features detected in one of the frames")
    
    # Match features using brute-force Hamming matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    
    # Sort by match quality (lower distance = better)
    matches = sorted(matches, key=lambda m: m.distance)
    
    # Keep top 50% of matches
    num_good = max(10, len(matches) // 2)
    matches = matches[:num_good]
    
    # Extract matched point coordinates
    ref_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    cur_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Estimate rigid transform (rotation + translation, no scaling/shearing)
    matrix, inliers = cv2.estimateAffinePartial2D(
        cur_pts, ref_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )
    
    if matrix is None:
        raise RuntimeError("Could not estimate transformation")
    
    return matrix


def warp_frame_affine(frame, matrix):
    """Apply an affine transformation matrix to warp a frame."""
    h, w = frame.shape[:2]
    warped = cv2.warpAffine(
        frame,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped

def apply_circular_crop(frame, radius_factor=1.0):
    """
    Apply a circular mask to remove border artefacts from rigid-body rotation.
    
    Parameters:
    -----------
    frame : np.ndarray
        The frame to crop
    radius_factor : float
        Scales the inscribed circle radius.
        1.0 = inscribed circle (fits within frame)
        <1.0 = smaller circle (more conservative)
        >1.0 = larger circle (keeps more data but may include artefacts)
    
    Returns:
    --------
    masked_frame : np.ndarray
        Frame with pixels outside the circle set to black
    
    Note:
    -----
    Adjust radius_factor if more peripheral tissue data is needed.
    Lower values are safer, higher values risk border artefacts.
    """
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    
    # Inscribed circle radius = half the shorter dimension
    base_radius = min(h, w) // 2
    radius = int(base_radius * radius_factor)
    
    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply mask to frame
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    
    return masked

def save_video(frames, output_path, fps=30):
    """
    Save a list of frames as a video file.
    
    Parameters:
    -----------
    frames : list of np.ndarray
        Frames to save
    output_path : str
        Path for the output video (should end in .mp4)
    fps : int
        Frames per second for the output video
    """
    if len(frames) == 0:
        raise ValueError("Cannot save empty frame list")
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    print(f"✓ Saved {len(frames)} frames to {output_path}")