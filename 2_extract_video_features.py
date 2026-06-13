import cv2
import os
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
# Use the PREPROCESSED videos from the previous step
INPUT_VIDEO_DIR = r"C:\Users\sanab\OneDrive\Desktop\Amakh+1\preprocessed_videos"
# Path to save the numerical skeletal features
OUTPUT_FEATURE_DIR = r"C:\Users\sanab\OneDrive\Desktop\Amakh+1\extracted_features\video"

ACTIONS = ['Violence', 'NonViolence']
# LSTM Sequence length (30 frames = 1 second of action at 30 FPS)
SEQUENCE_LENGTH = 30 

# Load the YOLOv8-Pose model (downloads automatically on first run)
pose_model = YOLO('yolov8n-pose.pt')

def extract_video_features():
    print("Starting Visual Feature Extraction (YOLOv8-Pose)...")

    for action in ACTIONS:
        input_dir = os.path.join(INPUT_VIDEO_DIR, action)
        output_dir = os.path.join(OUTPUT_FEATURE_DIR, action)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(input_dir):
            continue

        video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi'))]
        print(f"\nProcessing {len(video_files)} videos for: {action}")

        for v_file in video_files:
            video_path = os.path.join(input_dir, v_file)
            cap = cv2.VideoCapture(video_path)
            
            video_sequence = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Run YOLOv8-Pose on the frame
                results = pose_model(frame, verbose=False)[0]
                
                # 2. Extract Keypoints
                # YOLOv8-Pose returns 17 keypoints, each with (x, y, confidence)
                # Total features per frame: 17 * 3 = 51
                if results.keypoints is not None and len(results.keypoints.data) > 0:
                    # We take the first person detected in the frame
                    keypoints = results.keypoints.data[0].cpu().numpy().flatten()
                else:
                    # If no person is detected, fill with zeros to keep data shape consistent
                    keypoints = np.zeros(51)

                video_sequence.append(keypoints)

                # Optional: Stop if we have reached our sequence length 
                # (Or process whole video and segment later in training)
                if len(video_sequence) == SEQUENCE_LENGTH:
                    break

            cap.release()

            # 3. Ensure the sequence is exactly the right length (padding/truncating)
            if len(video_sequence) < SEQUENCE_LENGTH:
                # Pad with zeros if video is too short
                padding = [np.zeros(51) for _ in range(SEQUENCE_LENGTH - len(video_sequence))]
                video_sequence.extend(padding)
            else:
                video_sequence = video_sequence[:SEQUENCE_LENGTH]

            # 4. Save as .npy file
            feature_filename = os.path.splitext(v_file)[0] + ".npy"
            np.save(os.path.join(output_dir, feature_filename), np.array(video_sequence))
            print(f"  [Saved] {feature_filename}")

    print(f"\nVideo Extraction Complete. Features saved in: {OUTPUT_FEATURE_DIR}")

if __name__ == "__main__":
    extract_video_features()