import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

# --- 1. CONFIGURATION ---
METADATA_FILE = '../data/audio_metadata.csv'
OUTPUT_DIR = '../output/processed_data/'

# Feature Extraction Parameters
SAMPLE_RATE = 16000  # Target sample rate
N_MFCC = 40          # Number of MFCCs to extract
MAX_LEN = 174        # Fixed length for all feature vectors (in time steps)
# ---------------------------------------------------------------------------

def extract_features_from_file(file_path):
    """Loads an audio file with librosa and extracts MFCCs."""
    try:
        # Load audio file, automatically resampling to our target SAMPLE_RATE
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        
        # Pad or truncate to ensure all feature arrays have the same shape
        if mfccs.shape[1] > MAX_LEN:
            mfccs = mfccs[:, :MAX_LEN]
        else:
            pad_width = MAX_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return mfccs
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        # Return a matrix of zeros if a file is corrupted or unreadable
        return np.zeros((N_MFCC, MAX_LEN))

def process_and_save_features():
    """Main function to read metadata, extract features, and save them."""
    print("\n--- Building Audio Expert (Step 2): Extracting Audio Features ---")
    if not os.path.exists(METADATA_FILE):
        print(f"[ERROR] Metadata file '{METADATA_FILE}' not found. Please run the data preparation script first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = pd.read_csv(METADATA_FILE)
    
    all_audio_features = []
    all_labels = []

    # Loop through our clean metadata file
    for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing audio files"):
        full_path = row['full_audio_path']
        label = row['label']
        
        features = extract_features_from_file(full_path)
        all_audio_features.append(features)
        all_labels.append(label)

    # Convert lists to NumPy arrays
    X_audio = np.array(all_audio_features)
    y_audio = np.array(all_labels)

    # Save the processed data. This will overwrite the video .npy files,
    # which is what we want for training our audio model.
    np.save(os.path.join(OUTPUT_DIR, 'audio_features.npy'), X_audio)
    np.save(os.path.join(OUTPUT_DIR, 'audio_labels.npy'), y_audio)
    
    print("\n--- Audio Feature Extraction Complete ---")
    print(f"Audio features shape: {X_audio.shape}")
    print(f"Labels shape: {y_audio.shape}")
    print(f"Processed data saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    process_and_save_features()