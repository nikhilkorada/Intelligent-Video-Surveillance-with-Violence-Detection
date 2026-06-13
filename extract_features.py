import cv2
import os
import numpy as np
import librosa
from ultralytics import YOLO

# Load YOLOv8-Pose
pose_model = YOLO('yolov8n-pose.pt')

def extract_combined_features(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    video_features = []
    
    # 1. Extract Visual Pose (51 features per frame)
    while len(video_features) < 30:
        ret, frame = cap.read()
        if not ret: break
        
        results = pose_model(frame, verbose=False)
        if results[0].keypoints.data.shape[1] > 0:
            # Extract (x, y, conf) for 17 joints
            kpts = results[0].keypoints.data[0].cpu().numpy().flatten()
            if len(kpts) == 51:
                video_features.append(kpts)
    
    cap.release()
    
    # 2. Extract Audio MFCCs (40 features per 'frame')
    try:
        y, sr = librosa.load(video_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # Resize/Interpolate audio to match 30 video frames
        audio_features = cv2.resize(mfccs, (30, 40)).T # Shape (30, 40)
    except:
        audio_features = np.zeros((30, 40))

    if len(video_features) == 30:
        # Save separately for the Training Step
        np.save(os.path.join(save_path, 'video', os.path.basename(video_path) + '.npy'), np.array(video_features))
        np.save(os.path.join(save_path, 'audio', os.path.basename(video_path) + '.npy'), audio_features)

print("Feature Extraction script ready.")