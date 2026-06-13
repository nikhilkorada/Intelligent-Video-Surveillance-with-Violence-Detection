import os
import numpy as np
import tensorflow as tf
import librosa
import cv2
import sys

# --- 1. CONFIGURATION ---
VIDEO_MODEL_PATH = '../output/trained_model/video_violence_classifier.h5'
AUDIO_MODEL_PATH = '../output/trained_model/audio_violence_classifier.h5'

# --- HYPOTHETICAL NEW DATA ---
# IMPORTANT: Replace these with the FULL, ABSOLUTE paths to a real video and audio file.
# Example format: 'C:/Users/sanab/OneDrive/Desktop/Amakh+1/data/raw_video/Violence/V_2.mp4'
SAMPLE_VIDEO_PATH = r'C:\Users\sanab\OneDrive\Desktop\Amakh+1\content\Real Life Violence Dataset\Violence\V_3.mp4'
SAMPLE_AUDIO_PATH = r'C:\Users\sanab\OneDrive\Desktop\Amakh+1\data\Actor_20\03-01-08-02-01-02-20.wav'

# --- 2. LOAD MODELS AND PREPROCESSORS ---
print("--- Final Step: Running the Ensemble Predictor ---")

# Load the expert models
print("Loading Video and Audio Expert models...")
video_model = tf.keras.models.load_model(VIDEO_MODEL_PATH)
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)

# Load the video feature extractor model ONCE for efficiency
video_feature_extractor = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
print("Models loaded successfully.")

# --- 3. RE-USABLE PREPROCESSING FUNCTIONS ---

def preprocess_video(video_path):
    print(f"\nPreprocessing video: {os.path.basename(video_path)}...")
    IMG_SIZE = 224
    NUM_FRAMES = 16
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found at: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
    cap.release()
    
    if not frames: return None
        
    frames_np = np.array(frames)
    preprocessed_frames = tf.keras.applications.resnet50.preprocess_input(frames_np)
    frame_features = video_feature_extractor.predict(preprocessed_frames, verbose=0)
    return np.mean(frame_features, axis=0)

def preprocess_audio(audio_path):
    # (Audio function is the same as before)
    # ...
    # (The content of this function remains unchanged)
    print(f"Preprocessing audio: {os.path.basename(audio_path)}...")
    SAMPLE_RATE = 16000
    N_MFCC = 40
    MAX_LEN = 174
    
    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio file not found at: {audio_path}")
        return None
    
    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfccs = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    
    if mfccs.shape[1] > MAX_LEN:
        mfccs = mfccs[:, :MAX_LEN]
    else:
        pad_width = MAX_LEN - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    return mfccs
# --- 4. PROCESS THE SAMPLE DATA AND GET PREDICTIONS ---
# Process video
video_features = preprocess_video(SAMPLE_VIDEO_PATH)

# --- THIS IS THE FIX ---
# Add a check to ensure features were extracted successfully
if video_features is None:
    print("\nCould not process video. Exiting.")
    sys.exit() # Exits the script cleanly
# ---------------------

video_features = np.expand_dims(video_features, axis=0)
video_prediction = video_model.predict(video_features)[0][0]

# Process audio
audio_features = preprocess_audio(SAMPLE_AUDIO_PATH)
if audio_features is None:
    print("\nCould not process audio. Exiting.")
    sys.exit()

audio_features = np.expand_dims(audio_features, axis=0)
audio_features = np.transpose(audio_features, (0, 2, 1))
audio_prediction = audio_model.predict(audio_features)[0][0]

# --- 5. FUSE THE PREDICTIONS ---
print("\n--- Individual Expert Opinions ---")
print(f"Video Expert Prediction (Violence Probability): {video_prediction * 100:.2f}%")
print(f"Audio Expert Prediction (Violence Probability): {audio_prediction * 100:.2f}%")

ensemble_prediction = (video_prediction + audio_prediction) / 2

print("\n--- Final Ensemble Result ---")
print(f"Combined Violence Probability: {ensemble_prediction * 100:.2f}%")

if ensemble_prediction > 0.5:
    print("Final Verdict: Violence Detected")
else:
    print("Final Verdict: Non-Violence Detected")