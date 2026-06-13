import os
import numpy as np
import librosa
import cv2

# --- Configuration ---
DATA_PATH = 'data' # This contains Actor_01, Actor_02, etc.
SAVE_PATH = 'extracted_features/audio'
# RAVDESS Emotion mapping (Third digit in filename)
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
VIOLENCE_EMOTIONS = ['05', '06'] # Angry and Fearful as proxies for Violence

def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        # Standardize to 30 frames to match our LSTM input
        mfcc_resized = cv2.resize(mfcc, (30, 40)).T # Result: (30, 40)
        return mfcc_resized
    except Exception as e:
        return None

# --- Execution ---
print("Starting Audio Extraction from Actor folders...")

for actor_folder in os.listdir(DATA_PATH):
    actor_path = os.path.join(DATA_PATH, actor_folder)
    
    if os.path.isdir(actor_path) and actor_folder.startswith('Actor_'):
        print(f"Processing {actor_folder}...")
        
        for audio_file in os.listdir(actor_path):
            if audio_file.endswith('.wav'):
                # Filename example: 03-01-05-01-02-01-01.wav
                # The 3rd part is the emotion
                parts = audio_file.split('-')
                emotion = parts[2]
                
                label = 'Violence' if emotion in VIOLENCE_EMOTIONS else 'NonViolence'
                
                output_dir = os.path.join(SAVE_PATH, label)
                os.makedirs(output_dir, exist_ok=True)
                
                features = extract_mfcc(os.path.join(actor_path, audio_file))
                
                if features is not None:
                    # Save with a name that identifies the source
                    save_name = f"{actor_folder}_{audio_file}.npy"
                    np.save(os.path.join(output_dir, save_name), features)

print("\nSuccess! Audio features saved in extracted_features/audio")