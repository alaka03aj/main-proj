import cv2
import os
import argparse

def load_video(video_path):
    """Loads a video from the given path."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"Failed to open the video file: {video_path}")
    
    return video_capture

def extract_frames(video_capture, output_dir, frame_rate=1):
    """Extract frames from the video at a specified frame rate."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    saved_frame_count = 0
    success, frame = video_capture.read()

    while success:
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1
        success, frame = video_capture.read()

    video_capture.release()
    print(f"Extracted {saved_frame_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Frame Extraction")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the extracted frames.")
    parser.add_argument("--rate", type=int, default=1, help="Frame rate for extraction (e.g., 1 for every frame).")

    args = parser.parse_args()

    # Load the video file
    video_capture = load_video(args.video)

    # Extract frames
    extract_frames(video_capture, args.output, args.rate)
