import cv2
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import os

# --- 1. Setup & Load Model ---
# Load the 'Brain' you trained earlier
MODEL_PATH = 'violence_model.h5'
print("Loading Multi-Modal LSTM...")
model = load_model(MODEL_PATH)
pose_model = YOLO('yolov8n-pose.pt')

def select_file():
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    file_path = filedialog.askopenfilename(title="Select Video for Violence Detection",
                                         filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    root.destroy()
    return file_path

def extract_fused_features(video_path):
    """Processes a video file into a (1, 30, 91) tensor."""
    cap = cv2.VideoCapture(video_path)
    video_features = []
    
    print(f"Processing: {os.path.basename(video_path)}")
    
    # A. Visual Branch: Extract 30 frames of keypoints
    while len(video_features) < 30:
        ret, frame = cap.read()
        if not ret: break
        
        results = pose_model(frame, verbose=False)
        if len(results[0].keypoints.data) > 0:
            kpts = results[0].keypoints.data[0].cpu().numpy().flatten()
            if len(kpts) == 51:
                video_features.append(kpts)
    cap.release()

    if len(video_features) < 30:
        print("Error: Video too short or no person detected.")
        return None

    # B. Audio Branch: Extract 40 MFCCs
    try:
        audio, sr = librosa.load(video_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        # Resize to match 30 frames
        audio_features = cv2.resize(mfcc, (30, 40)).T # (30, 40)
    except:
        print("Warning: Audio extraction failed. Using silence.")
        audio_features = np.zeros((30, 40))

    # C. Late Fusion: Combine into (30, 91)
    fused = np.concatenate([np.array(video_features), audio_features], axis=1)
    return np.expand_dims(fused, axis=0) # Shape (1, 30, 91)

# --- 2. Main Execution ---
video_path = select_file()

if video_path:
    input_tensor = extract_fused_features(video_path)
    
    if input_tensor is not None:
        # Prediction: [Violence_Prob, NonViolence_Prob]
        prediction = model.predict(input_tensor, verbose=0)
        violence_score = prediction[0][0]
        non_violence_score = prediction[0][1]
        
        print("\n" + "="*30)
        print(f"RESULT FOR: {os.path.basename(video_path)}")
        print(f"Violence Probability: {violence_score*100:.2f}%")
        print(f"Non-Violence Probability: {non_violence_score*100:.2f}%")
        
        if violence_score > 0.5:
            print("FINAL VERDICT: ⚠️ VIOLENCE DETECTED")
        else:
            print("FINAL VERDICT: ✅ NORMAL BEHAVIOR")
        print("="*30)
else:
    print("No file selected.")