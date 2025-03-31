import cv2
import os
import argparse

def initialize_camera(camera_index=0):
    """Initializes the camera and returns the capture object."""
    
    print(f"Attempting to initialize camera with index {camera_index}...")
    video_capture = cv2.VideoCapture(camera_index)
    if not video_capture.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        raise ValueError(f"Failed to open the camera feed at index {camera_index}.")
    print(f"Camera initialized successfully!")
    return video_capture

def extract_frames_from_camera(video_capture, output_dir, frame_rate=1):
    """Extract frames from live camera feed at a specified frame rate."""
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist. Creating...")
        os.makedirs(output_dir)

    frame_count = 0
    saved_frame_count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            print("Error: Failed to grab frame from camera.")
            break

        # Save frame at the specified rate
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            print(f"Frame saved: {frame_filename}")

        frame_count += 1

        # Show the live feed in a window
        cv2.imshow("Live Camera Feed", frame)

        # Press 'q' to quit the feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting on user request (pressed 'q').")
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print(f"Extracted {saved_frame_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Camera Feed Frame Extraction")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the extracted frames.")
    parser.add_argument("--rate", type=int, default=1, help="Frame rate for extraction (e.g., 1 for every frame).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default is 0 for the default camera).")

    args = parser.parse_args()

    # Initialize the camera
    try:
        video_capture = initialize_camera(args.camera)
    except ValueError as e:
        print(e)
        exit(1)

    # Extract frames from the live feed
    extract_frames_from_camera(video_capture, args.output, args.rate)

