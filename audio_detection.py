import os
import numpy as np
import librosa
import tensorflow as tf
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Path to your audio file (can be .mp3, .wav, or even a .mp4 video)
INPUT_AUDIO_PATH = "path/to/your/audio_file.wav"
AUDIO_MODEL_PATH = '../output/trained_model/audio_violence_classifier.h5'

# Model Constants (Matches your training setup)
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 174 

def preprocess_audio(file_path):
    """Loads and converts audio to MFCC features."""
    try:
        # librosa can load audio directly from video files too
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Extract Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        
        # Pad or truncate to match model input shape
        if mfccs.shape[1] > MAX_LEN:
            mfccs = mfccs[:, :MAX_LEN]
        else:
            mfccs = np.pad(mfccs, ((0, 0), (0, MAX_LEN - mfccs.shape[1])), mode='constant')
        
        # Reshape for model (batch, time, features)
        return np.transpose(np.expand_dims(mfccs, axis=0), (0, 2, 1))
    
    except Exception as e:
        print(f"[ERROR] Loading audio: {e}")
        return None

def main():
    print("--- Standalone Audio Violence Detection ---")
    
    # Load your pre-trained audio model
    if not os.path.exists(AUDIO_MODEL_PATH):
        print(f"[EXIT] Model file not found at {AUDIO_MODEL_PATH}")
        return
        
    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
    
    # Preprocess
    print(f"Analyzing: {os.path.basename(INPUT_AUDIO_PATH)}...")
    features = preprocess_audio(INPUT_AUDIO_PATH)
    
    if features is not None:
        # Prediction
        prediction = audio_model.predict(features, verbose=0)[0][0]
        confidence = prediction * 100
        
        print("\n--- RESULTS ---")
        print(f"Violence Probability: {confidence:.2f}%")
        
        if prediction > 0.5:
            print("Verdict: 🚨 VIOLENCE DETECTED (Audio-based)")
        else:
            print("Verdict: ✅ Normal Environment")
    else:
        print("Could not process the audio file.")

if __name__ == '__main__':
    main()