import cv2
import os
import argparse
import logging
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def setup_logging():
    """Sets up logging for the script."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")


def initialize_camera(camera_index=0):
    """Initializes the camera and returns the capture object."""
    logging.info(f"Attempting to initialize camera with index {camera_index}...")
    video_capture = cv2.VideoCapture(camera_index)
    if not video_capture.isOpened():
        raise ValueError(f"Error: Could not open camera with index {camera_index}.")
    logging.info("Camera initialized successfully!")
    return video_capture


def resize_frame(frame, width=640, height=480):
    """Resizes the frame to the given dimensions."""
    return cv2.resize(frame, (width, height))


def is_key_frame(prev_frame, current_frame, threshold=0.5):
    """Determines if the current frame is a key frame based on SSIM."""
    if prev_frame is None:
        return True  # Always treat the first frame as a key frame

    # Convert frames to grayscale for SSIM calculation
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two frames
    score, _ = compare_ssim(gray_prev, gray_curr, full=True)
    logging.debug(f"SSIM score: {score}")
    return score < threshold  # Lower SSIM indicates higher difference


def extract_frames(video_capture, output_dir, frame_rate=1, resize_dims=(640, 480), key_frame_threshold=0.5):
    """Extract frames from live camera feed."""
    if not os.path.exists(output_dir):
        logging.info(f"Output directory '{output_dir}' does not exist. Creating...")
        os.makedirs(output_dir)

    frame_count = 0
    saved_frame_count = 0
    prev_frame = None

    while True:
        success, frame = video_capture.read()
        if not success:
            logging.error("Failed to grab frame from camera.")
            break

        # Resize frame
        if resize_dims:
            frame = resize_frame(frame, width=resize_dims[0], height=resize_dims[1])

        # Save frame at specified frame rate or if it's a key frame
        if frame_count % frame_rate == 0 or is_key_frame(prev_frame, frame, threshold=key_frame_threshold):
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            logging.info(f"Frame saved: {frame_filename}")
            prev_frame = frame  # Update the previous frame

        frame_count += 1

        # Show the live feed in a window
        cv2.imshow("Live Camera Feed", frame)

        # Press 'q' to quit the feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting on user request (pressed 'q').")
            break

    video_capture.release()
    cv2.destroyAllWindows()
    logging.info(f"Extracted {saved_frame_count} frames to {output_dir}")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Live Camera Feed Frame Extraction with Sampling and Resizing")
    parser.add_argument("--output", type=str, default="frames", help="Directory to save the extracted frames.")
    parser.add_argument("--rate", type=int, default=1, help="Frame rate for extraction (e.g., 1 for every frame).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default is 0 for the default camera).")
    parser.add_argument("--resize", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help="Resize dimensions for the frames, e.g., --resize 640 480.")
    parser.add_argument("--key-threshold", type=float, default=0.5,
                        help="Threshold for key frame detection based on SSIM. Lower values are stricter.")
    return parser.parse_args()


def main():
    """Main function to control the script execution."""
    setup_logging()

    args = parse_args()

    # Initialize the camera
    try:
        video_capture = initialize_camera(args.camera)
    except ValueError as e:
        logging.error(e)
        exit(1)

    # Extract frames from the live feed
    extract_frames(video_capture, args.output, frame_rate=args.rate, 
                   resize_dims=tuple(args.resize) if args.resize else None,
                   key_frame_threshold=args.key_threshold)


if __name__ == "__main__":
    main()
