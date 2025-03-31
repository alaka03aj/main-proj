import cv2
import numpy as np
import torch
import os

class StampedePreprocessor:
    def __init__(self, video_path, output_dir, sample_rate=5):
        self.video_path = video_path
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        os.makedirs(output_dir, exist_ok=True)

    def extract_frames(self, video_capture):
        """Extract frames with sampling"""
        frames = []
        frame_count = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_count % self.sample_rate == 0:  # Sample every Nth frame
                frames.append(frame)

            frame_count += 1

        video_capture.release()
        print(f"Extracted {len(frames)} frames (Sampled every {self.sample_rate} frames).")
        return frames

    def extract_key_frames_bg_subtraction(self, frames, output_dir):
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        def process_frame(frame):
            fg_mask = bg_subtractor.apply(frame)
            motion_area = np.sum(fg_mask > 0)
            return motion_area

        initial_motion = [process_frame(f) for f in frames[:10]]
        average_motion = np.mean(initial_motion)
        threshold = average_motion * 0.005
        print(f"Motion threshold: {threshold}")

        key_frames = [frames[0]]
        prev_motion_area = process_frame(frames[0])
        for i in range(1, len(frames)):
            curr_motion_area = process_frame(frames[i])
            if (abs(curr_motion_area - prev_motion_area) > threshold):
                key_frames.append(frames[i])
            prev_motion_area = curr_motion_area

        self.save_intermediate_frames(key_frames, output_dir, "key_frame")
        print(f"Extracted {len(key_frames)} key frames.")
        return key_frames

    def save_intermediate_frames(self, frames, output_dir, prefix):
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame_path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
            if frame.dtype == np.float32:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(frame_path, frame)

    def reduce_noise(self, frames, method='bilateral', output_dir=None):
        """Apply noise reduction to frames"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        denoised_frames = []
        for frame in frames:
            if method == "gaussian":
                # Gaussian Blur
                denoised = cv2.GaussianBlur(frame, (5, 5), 0)
            elif method == "bilateral":
                # Bilateral Filter (edge-preserving)
                denoised = cv2.bilateralFilter(frame, 5, 45, 45)
            else:
                raise ValueError("Unsupported noise reduction method.")
            
            denoised_frames.append(denoised)
        
        # Save denoised frames if output directory is provided
        if output_dir:
            for i, frame in enumerate(denoised_frames):
                cv2.imwrite(os.path.join(output_dir, f'denoised_frame_{i:04d}.jpg'), frame)
        
        return denoised_frames

    def preprocess_for_detection(self, frames, target_size=(640, 640)):
        """Common preprocessing for both Mask R-CNN and YOLOv5"""
        processed_frames = []

        for frame in frames:
            # Resize while maintaining aspect ratio
            h, w = frame.shape[:2]
            scale = min(target_size[0] / h, target_size[1] / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(frame, (new_w, new_h))

            # Pad to target size
            top_pad = (target_size[0] - new_h) // 2
            bottom_pad = target_size[0] - new_h - top_pad
            left_pad = (target_size[1] - new_w) // 2
            right_pad = target_size[1] - new_w - left_pad
            padded = cv2.copyMakeBorder(resized, top_pad, bottom_pad, left_pad, right_pad,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Normalize by subtracting dataset mean (crowd counting dataset mean)
            dataset_mean = [105.14115322735742, 98.00561142108008, 96.10478742777283]
            normalized = padded.astype(np.float32) / 255.0 - np.array(dataset_mean) / 255.0

            processed_frames.append(normalized)

        return processed_frames

    def save_processed_frames(self, frames, model_type):
        """Save processed frames"""
        output_path = os.path.join(self.output_dir, model_type)
        os.makedirs(output_path, exist_ok=True)

        for i, frame in enumerate(frames):
            # Save the frame directly
            cv2.imwrite(os.path.join(output_path, f'{model_type}_frame_{i:04d}.jpg'), frame)

    def process(self, noise_method='bilateral'):
        """Main processing pipeline"""
        # Capture video
        cap = cv2.VideoCapture(self.video_path)

        # Extract frames with sampling
        frames = self.extract_frames(cap)

        # Extract key frames using background subtraction
        key_frames = self.extract_key_frames_bg_subtraction(
            frames,
            os.path.join(self.output_dir, 'key_frames')
        )

        # Reduce noise
        denoised_frames = self.reduce_noise(
            key_frames, 
            method=noise_method, 
            output_dir=os.path.join(self.output_dir, 'denoised_frames')
        )

        # Preprocess frames
        # processed_frames = self.preprocess_for_detection(denoised_frames)

        # Save processed frames
        self.save_processed_frames(denoised_frames, 'preprocessed')

        return denoised_frames

def main():
    # You can now specify the sample rate when creating the preprocessor
    preprocessor = StampedePreprocessor(
        'video.mp4', 
        'output_preprocessed', 
        sample_rate=5  # Extract every 5th frame
    )
    final_processed_frames = preprocessor.process(noise_method='bilateral')

if __name__ == "__main__":
    main()