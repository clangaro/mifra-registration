"""
Main script: run full registration pipeline on a video.
Usage: python main.py
"""

from register_frames import load_video, register_all_frames, save_video 

# configuration
INPUT_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/data/MVI_6805.MP4"
OUTPUT_VIDEO = "/Users/carolinalangaro/Desktop/mifra_registration/output-videos/registered_video.mp4"
REFERENCE_INDEX = 0  # index of frame to use as reference for registration
FPS = 30

def main():
    print(f"Loading video: {INPUT_VIDEO}")
    frames = load_video(INPUT_VIDEO)
    print(f"Loaded {len(frames)} frames.")

    print("Registering all frames (this may take a while)...")
    registered_frames = register_all_frames(frames, reference_index=REFERENCE_INDEX, verbose=True)

    print(f"Saving registered video to: {OUTPUT_VIDEO}")
    save_video(registered_frames, OUTPUT_VIDEO, fps=FPS)
    print("Done!")

if __name__ == "__main__":
    main()