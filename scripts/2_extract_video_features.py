import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

# --- 1. CONFIGURATION ---
METADATA_FILE = r'C:\Users\sanab\OneDrive\Desktop\Amakh+1\data\video_metadata.csv'
OUTPUT_DIR = r'C:\Users\sanab\OneDrive\Desktop\Amakh+1\output\processed_data'

# Feature Extraction Parameters
IMG_SIZE = 224
NUM_FRAMES = 16 # Number of frames to sample from each video
# ---------------------------------------------------------------------------


# --- 2. PRE-LOAD THE CNN MODEL (ResNet50) ---
# This is done once to avoid reloading the model for every video.
# ---------------------------------------------------------------------------
print("Loading pre-trained ResNet50 model for feature extraction...")
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
base_model.trainable = False # We use it as a fixed feature extractor
print("Model loaded successfully.")
# ---------------------------------------------------------------------------


def extract_video_features(video_path):
    """
    Reads a video file, samples a fixed number of frames, and extracts deep
    features from them using the pre-trained ResNet50 model.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Select evenly spaced frame indices to sample
        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert frame from BGR (OpenCV default) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame to the size expected by ResNet50
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frames.append(frame)
        cap.release()
        
        if not frames:
            # If no frames were read, return a vector of zeros
            return np.zeros(base_model.output_shape[1])
            
        # Convert list of frames to a single numpy array
        frames_np = np.array(frames)
        
        # Preprocess the frames for the ResNet50 model
        preprocessed_frames = tf.keras.applications.resnet50.preprocess_input(frames_np)
        
        # Extract features using the pre-trained model
        frame_features = base_model.predict(preprocessed_frames, verbose=0)
        
        # Aggregate the features of all frames by taking the mean
        video_feature_vector = np.mean(frame_features, axis=0)
        
        return video_feature_vector
    except Exception as e:
        print(f"\nError processing video {video_path}: {e}")
        # Return a zero vector in case of a critical error
        return np.zeros(base_model.output_shape[1])


def process_and_save_features():
    """
    Main function to read metadata, loop through videos, extract features, and save them.
    """
    print("\n--- Building Video Expert (Step 2): Extracting Video Features ---")
    if not os.path.exists(METADATA_FILE):
        print(f"[ERROR] Metadata file not found at {METADATA_FILE}. Please run the previous script first.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load the metadata
    metadata = pd.read_csv(METADATA_FILE)
    
    all_video_features = []
    all_labels = []

    # Loop through each video in the metadata file with a progress bar
    for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting features"):
        video_path = row['video_path']
        label = row['label']
        
        # Extract features for the current video
        features = extract_video_features(video_path)
        
        all_video_features.append(features)
        all_labels.append(label)

    # Convert the lists of features and labels to numpy arrays
    X_video = np.array(all_video_features)
    y_video = np.array(all_labels)

    # Save the processed data to disk
    np.save(os.path.join(OUTPUT_DIR, 'video_features.npy'), X_video)
    np.save(os.path.join(OUTPUT_DIR, 'video_labels.npy'), y_video)
    
    print("\n--- Feature Extraction Complete ---")
    print(f"Video features shape: {X_video.shape}")
    print(f"Labels shape: {y_video.shape}")
    print(f"Processed data saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    process_and_save_features()