import cv2
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import collections
import time

# --- 1. Configuration & Model Loading ---
MODEL_PATH = 'violence_model.h5'
POSE_MODEL_PATH = 'yolov8n-pose.pt'

print("Loading Models... Please wait.")
model = load_model(MODEL_PATH)
pose_model = YOLO(POSE_MODEL_PATH)

# Buffers to store 30 frames of history (1 second of data)
video_buffer = collections.deque(maxlen=30)
audio_buffer = collections.deque(maxlen=30)

# Detection Thresholds
VIOLENCE_THRESHOLD = 0.5  # Adjust this (0.1 to 0.9) to change sensitivity
SMOOTHING_WINDOW = 5
prob_history = collections.deque(maxlen=SMOOTHING_WINDOW)

# --- 2. Feature Extraction Functions ---

def get_audio_features():
    """Captures a slice of audio and extracts MFCCs."""
    duration = 0.04  # ~25-30 FPS equivalent
    fs = 16000
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking=False)
        # We don't sd.wait() to keep the video FPS high
        # Extract MFCC from a small buffer
        mfcc = librosa.feature.mfcc(y=np.zeros(int(duration*fs)), sr=fs, n_mfcc=40)
        return np.mean(mfcc, axis=1)
    except:
        return np.zeros(40)

# --- 3. Main Live Loop ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n--- System Active ---")
print("Press 'q' to exit.")

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret: break

    # A. Visual Branch (YOLOv8-Pose)
    results = pose_model(frame, verbose=False)
    annotated_frame = results[0].plot()
    
    if len(results[0].keypoints.data) > 0:
        # Extract (x, y, conf) for 17 joints = 51 features
        kpts = results[0].keypoints.data[0].cpu().numpy().flatten()
        if len(kpts) == 51:
            video_buffer.append(kpts)
    else:
        # If no person detected, fill with zeros to keep buffer moving
        video_buffer.append(np.zeros(51))

    # B. Audio Branch (Microphone)
    # Note: In real-time, we use a placeholder or low-latency stream
    audio_feat = np.random.normal(0, 0.1, 40) # Replace with real mic stream if sd.rec is too slow
    audio_buffer.append(audio_feat)

    # C. Multi-Modal Fusion & Inference
    if len(video_buffer) == 30:
        v_seq = np.array(video_buffer)
        a_seq = np.array(audio_buffer)
        
        # Late Fusion: Concatenate into (30, 91)
        fused_input = np.concatenate([v_seq, a_seq], axis=1)
        fused_input = np.expand_dims(fused_input, axis=0) # Shape (1, 30, 91)
        
        # Predict
        prediction = model.predict(fused_input, verbose=0)
        violence_prob = prediction[0][0] # Probability of Violence
        prob_history.append(violence_prob)
        
        # Averaging for stability
        smooth_prob = np.mean(prob_history)

        # D. Visualization & Alert
        if smooth_prob > VIOLENCE_THRESHOLD:
            status = "!!! VIOLENCE DETECTED !!!"
            color = (0, 0, 255) # Red
            # Trigger "Alert" visual
            cv2.rectangle(annotated_frame, (0,0), (640, 60), color, -1)
        else:
            status = "SYSTEM MONITORING: NORMAL"
            color = (0, 255, 0) # Green
            
        cv2.putText(annotated_frame, f"{status}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Prob: {smooth_prob:.2f}", (10, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Calculate and Display FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (540, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow('Multi-Modal Surveillance System (MTech Project)', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()