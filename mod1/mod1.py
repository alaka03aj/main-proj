import cv2
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.utils import normalize
import os
from torchvision import transforms
from skimage.metrics import structural_similarity as compare_ssim
import imageio  # Using imageio for frame saving

# --- Step 1: Capture Pre-Recorded Video ---
def capture_video(video_path="video.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file {video_path}.")
    return cap

# --- Step 2: Convert Video Frames into Image Frames ---
def extract_frames(video_capture, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append(frame)
        frame_count += 1
    video_capture.release()
    print(f"Extracted {frame_count} frames.")
    return frames

# --- Step 3: Perform Selective Frame Sampling Using PyTorch ---
def sample_frames(frames, sampling_rate):
    return frames[::sampling_rate]

# --- Step 4: Extract Key Frames Using Optical Flow ---
def extract_key_frames(frames, threshold=0.5):
    key_frames = [frames[0]]
    prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.linalg.norm(flow, axis=2).mean()
        if magnitude > threshold:
            key_frames.append(frames[i])
        prev_frame = curr_frame
    print(f"Extracted {len(key_frames)} key frames.")
    return key_frames

# --- Step 5: Resize Frames ---
def resize_frames(frames, size=(512, 512)):
    return [cv2.resize(frame, size) for frame in frames]

# --- Step 6: Normalize Frames Using TensorFlow ---
def normalize_frames(frames):
    return [normalize(frame.astype('float32')) for frame in frames]

# --- Step 7: Apply Noise Reduction ---
def reduce_noise(frames, method="gaussian"):
    if method == "gaussian":
        return [cv2.GaussianBlur(frame, (5, 5), 0) for frame in frames]
    elif method == "bilateral":
        return [cv2.bilateralFilter(frame, 9, 75, 75) for frame in frames]
    else:
        raise ValueError("Unsupported noise reduction method.")

# --- Step 8: Implement Skip Connections Using PyTorch ---
class SkipConnections(torch.nn.Module):
    def __init__(self):
        super(SkipConnections, self).__init__()
        self.layer1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.skip = torch.nn.Conv2d(3, 128, kernel_size=1)

    def forward(self, x):
        out1 = torch.relu(self.layer1(x))
        out2 = torch.relu(self.layer2(out1))
        skip_out = self.skip(x)
        return out2 + skip_out

def apply_skip_connections(frames):
    model = SkipConnections()
    preprocess = transforms.Compose([transforms.ToTensor()])
    processed_frames = []
    for frame in frames:
        # Handle single-pixel frames by resizing to minimum size (512x512)
        if frame.shape == (1, 1):  # Check for single-pixel grayscale frame
            frame = cv2.resize(frame, (512, 512))

        tensor_frame = preprocess(frame).unsqueeze(0)  # Add batch dimension
        processed_frame = model(tensor_frame)

        # Convert tensor back to NumPy array
        processed_frame = processed_frame.squeeze(0).detach().numpy()

        # Convert grayscale to RGB if necessary
        if processed_frame.ndim == 2:
            processed_frame = np.stack([processed_frame] * 3, axis=-1)

        processed_frames.append(processed_frame)
    return processed_frames

# --- Step 9: Save Preprocessed Frames ---
def save_frames_with_imageio(frames, output_dir): 
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"preprocessed_frame_{i:04d}.jpg")
        if frame.shape[2] == 1:  # Handle single-channel frames
            frame = np.tile(frame, (1, 3, 1))  # Replicate channel for RGB
        imageio.imwrite(frame_path, frame)  # Use imageio for all frames
    print(f"Saved {len(frames)} preprocessed frames.")

# --- Main Function ---
def main(video_path, output_dir, sampling_rate=5, noise_method="gaussian"):
    cap = capture_video(video_path)
    frames = extract_frames(cap, os.path.join(output_dir, "raw_frames"))
    sampled_frames = sample_frames(frames, sampling_rate)
    key_frames = extract_key_frames(sampled_frames)
    resized_frames = resize_frames(key_frames)
    normalized_frames = normalize_frames(resized_frames)
    denoised_frames = reduce_noise(normalized_frames, method=noise_method)
    processed_frames = apply_skip_connections(denoised_frames)
    save_frames_with_imageio(processed_frames, os.path.join(output_dir, "preprocessed_frames")) 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-recorded Video Preprocessing Pipeline")
    parser.add_argument("--video-path", type=str, required=True, help="Path to the pre-recorded video file.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed frames.")
    parser.add_argument("--sampling-rate", type=int, default=1, help="Frame sampling rate.")
    parser.add_argument("--noise-method", type=str, default="gaussian", help="Noise reduction method (gaussian or bilateral).")
    args = parser.parse_args()

    main(args.video_path, args.output, args.sampling_rate, args.noise_method)