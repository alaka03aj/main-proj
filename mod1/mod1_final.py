import cv2
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.utils import normalize #type: ignore
import os
from torchvision import transforms

# --- Utility Function: Save Intermediate Frames ---

def save_intermediate_frames(frames, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
        if frame.dtype == np.float32:  # Convert to uint8 if normalized
            frame = (frame * 255).astype(np.uint8)
        cv2.imwrite(frame_path, frame)


# --- Step 1: Capture Video ---
def capture_video(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video source {source}.")
    return cap

# --- Step 2: Convert Video Frames into Image Frames ---
def extract_frames(video_capture, output_dir):
    frames = []
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    video_capture.release()
    save_intermediate_frames(frames, output_dir, "raw_frame")
    print(f"Extracted {frame_count} frames.")
    return frames

# --- Step 3: Perform Selective Frame Sampling ---
def sample_frames(frames, sampling_rate, output_dir):
    sampled_frames = frames[::sampling_rate]
    save_intermediate_frames(sampled_frames, output_dir, "sampled_frame")
    return sampled_frames

# --- Step 4: Extract Key Frames Using Optical Flow ---
def extract_key_frames(frames, threshold, output_dir):
    key_frames = [frames[0]]
    prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.linalg.norm(flow, axis=2).mean()
        if magnitude > threshold:
            key_frames.append(frames[i])
        prev_frame = curr_frame
    save_intermediate_frames(key_frames, output_dir, "key_frame")
    print(f"Extracted {len(key_frames)} key frames.")
    return key_frames

def extract_key_frames_optimised(frames, threshold, output_dir):
  from concurrent.futures import ThreadPoolExecutor

  def process_frame_pair(frame_pair):
    prev,curr = frame_pair
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    #diff = cv2.absdiff(prev_gray, curr_gray)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.linalg.norm(flow, axis=2).mean()
    return magnitude > threshold

  frame_pairs = list(zip(frames[:-1], frames[1:]))
  with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_frame_pair, frame_pairs))

  key_frames = [frames[0]]

  for i,is_key in enumerate(results):
    if is_key:
      key_frames.append(frames[i+1])

  save_intermediate_frames(key_frames, output_dir, "key_frame")
  print(f"Extracted {len(key_frames)} key frames.")
  return key_frames

def extract_key_frames_bg_subtraction(frames, output_dir):
  bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

  def process_frame(frame):
    fg_mask = bg_subtractor.apply(frame)
    motion_area = np.sum(fg_mask > 0)
    return motion_area

  initial_motion = [process_frame(f) for f in frames[:10]]
  average_motion = np.mean(initial_motion)
  threshold = average_motion * 0.005
  print(threshold)

  key_frames = [frames[0]]
  prev_motion_area = process_frame(frames[0])
  for i in range(1, len(frames)):
    curr_motion_area = process_frame(frames[i])
    if (abs(curr_motion_area - prev_motion_area) > threshold):
      key_frames.append(frames[i])
    prev_motion_area = curr_motion_area

  save_intermediate_frames(key_frames, output_dir, "key_frame")
  print(f"Extracted {len(key_frames)} key frames.")
  return key_frames

# --- Step 5: Resize Frames ---
def resize_frames(frames, size, output_dir):
    resized_frames = [cv2.resize(frame, size) for frame in frames]
    save_intermediate_frames(resized_frames, output_dir, "resized_frame")
    return resized_frames

# --- Step 6: Normalize Frames ---
# def normalize_frames(frames, output_dir):
#     normalized_frames = [normalize(frame.astype('float32')) for frame in frames]
#     save_intermediate_frames(normalized_frames, output_dir, "normalized_frame")
#     return normalized_frames

def normalize_frames(frames, output_dir):
    normalized_frames = [(frame - frame.min()) / (frame.max() - frame.min() + 1e-8) for frame in frames]  # Avoid division by zero
    save_intermediate_frames(normalized_frames, output_dir, "normalized_frame")
    return normalized_frames


# --- Step 7: Apply Noise Reduction ---
def reduce_noise(frames, method, output_dir):
    if method == "gaussian":
        denoised_frames = [cv2.GaussianBlur(frame, (5, 5), 0) for frame in frames]
    elif method == "bilateral":
        denoised_frames = [cv2.bilateralFilter(frame, 5, 45, 45) for frame in frames]
    else:
        raise ValueError("Unsupported noise reduction method.")
    save_intermediate_frames(denoised_frames, output_dir, "denoised_frame")
    return denoised_frames


# --- Step 8: Implement Skip Connections ---
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import normalize

# class SkipConnections(tf.keras.Model):
#     def __init__(self):
#         super(SkipConnections, self).__init__()
#         self.layer1 = Conv2D(64, kernel_size=3, padding='same', activation=None)
#         self.layer2 = Conv2D(128, kernel_size=3, padding='same', activation=None)
#         self.skip = Conv2D(128, kernel_size=1, padding='same', activation=None)

#     def call(self, inputs):
#         out1 = tf.nn.relu(self.layer1(inputs))
#         out2 = tf.nn.relu(self.layer2(out1))
#         skip_out = self.skip(inputs)
#         return out2 + skip_out

import tensorflow as tf
import numpy as np


class SkipConnections(tf.keras.Model):
    def __init__(self):
        super(SkipConnections, self).__init__()  # Properly initialize the parent class

        initializer = tf.keras.initializers.GlorotUniform(seed=42)  # Ensure deterministic results

        # Define layers
        self.layer1= tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation=None, kernel_initializer=initializer)
        self.layer2 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation=None, kernel_initializer=initializer)
        self.skip = tf.keras.layers.Conv2D(128, kernel_size=1, padding='same', activation=None, kernel_initializer=initializer)

    def build(self, input_shape):
        """Ensures model weights are built properly before first call."""
        super().build(input_shape)

    def call(self, inputs, training=False):
        out1 = tf.nn.relu(self.layer1(inputs))
        out2 = tf.nn.relu(self.layer2(out1))
        skip_out = self.skip(inputs)
        return out2 + skip_out  # Element-wise addition



# def save_intermediate_frames(frames, output_dir, prefix):
#     os.makedirs(output_dir, exist_ok=True)
#     for i, frame in enumerate(frames):
#         frame_path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
#         frame_uint8 = (frame * 255).astype(np.uint8)  # Convert normalized frames to uint8
#         tf.keras.preprocessing.image.save_img(frame_path, frame_uint8)

def save_intermediate_frames(frames, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
        frame_uint8 = np.clip(frame * 255, 0, 255).astype(np.uint8)  # Ensure values are in range
        cv2.imwrite(frame_path,frame_uint8)

def apply_skip_connections(frames, output_dir):
    model = SkipConnections()
    model.build(input_shape=(None, 400, 600, 3))
    processed_frames = []

    for frame in frames:
        tensor_frame = tf.convert_to_tensor(frame, dtype=tf.float32)  # Convert to Tensor
        tensor_frame = tf.expand_dims(tensor_frame, axis=0)  # Add batch dimension
        processed_frame = model(tensor_frame)
        processed_frame = tf.squeeze(processed_frame, axis=0).numpy()  # Remove batch dimension

        # Reduce channels to 3 (RGB) or 1 (Grayscale)
        # Option 1: Use only the first 3 channels
        processed_frame = processed_frame[:, :, :3]
        # Option 2: Average across all channels to create a grayscale image
        # processed_frame = np.mean(processed_frame, axis=-1, keepdims=True)

        # Normalize the frame
        processed_frame = normalize(processed_frame)
        processed_frames.append(processed_frame)

    save_intermediate_frames(processed_frames, output_dir, "skip_frame")
    return processed_frames

# --- Main Function ---
def main(source, output_dir, noise_method, sampling_rate=5):
    import tensorflow as tf
    import numpy as np
    import random
    import os

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'

    # Force TensorFlow to use deterministic algorithms
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    cap = capture_video(source)

    raw_output_dir = os.path.join(output_dir, "raw_frames")
    sampled_output_dir = os.path.join(output_dir, "sampled_frames")
    key_output_dir = os.path.join(output_dir, "key_frames")
    resized_output_dir = os.path.join(output_dir, "resized_frames")
    normalized_output_dir = os.path.join(output_dir, "normalized_frames")
    denoised_output_dir = os.path.join(output_dir, "denoised_frames")
    skip_output_dir = os.path.join(output_dir, "skip_frames")

    frames = extract_frames(cap, raw_output_dir)
    sampled_frames = sample_frames(frames, sampling_rate, sampled_output_dir)
    key_frames = extract_key_frames_bg_subtraction(sampled_frames, output_dir=key_output_dir)
    resized_frames = resize_frames(key_frames, size=(600, 400), output_dir=resized_output_dir)
    denoised_frames = reduce_noise(resized_frames, method=noise_method, output_dir=denoised_output_dir)
    normalized_frames = normalize_frames(denoised_frames, output_dir=normalized_output_dir)
    processed_frames = apply_skip_connections(normalized_frames, output_dir=skip_output_dir)

    final_output_dir = os.path.join(output_dir, "preprocessed_frames")
    save_intermediate_frames(processed_frames, final_output_dir, "preprocessed_frame")

if __name__ == "__main__":
    main("video.mp4", "output", "bilateral", 5)