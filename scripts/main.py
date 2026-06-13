import os
import sys
import subprocess
import numpy as np
import tensorflow as tf
import librosa
import cv2
import requests # For sending Discord notifications
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# --- 1. CONFIGURATION ---
INPUT_VIDEO_PATH = r"C:\Users\sanab\OneDrive\Desktop\Amakh+1\test\From KlickPin CF اعـِٰ̲ــٰـ͢ــٰٖتـــِٰ̲زاެެل🚸†⤿ _ Funny baby gif Cute couple cartoon Photo poses for boy.mp4"

# Load Discord Webhook URL securely from the .env file
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

VIDEO_MODEL_PATH = '../output/trained_model/video_violence_classifier.h5'
AUDIO_MODEL_PATH = '../output/trained_model/audio_violence_classifier.h5'

# --- 2. NOTIFICATION FUNCTION (DISCORD) ---
def send_discord_notification(final_verdict, confidence_score):
    """Sends a notification to a Discord channel using a Webhook."""
    if not DISCORD_WEBHOOK_URL:
        print("\n[WARNING] DISCORD_WEBHOOK_URL not found in .env file. Skipping notification.")
        return
    
    message = f"🚨 **ALERT: {final_verdict}** 🚨\n> **Confidence Score:** `{confidence_score * 100:.2f}%`\n> **File:** `{os.path.basename(INPUT_VIDEO_PATH)}`"
    data = {"content": message, "username": "Violence Detection Bot"}
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        response.raise_for_status()
        print("\nDiscord notification sent successfully!")
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Failed to send Discord notification: {e}")

# --- 3. PREPROCESSING & MAIN LOGIC ---
def extract_audio(video_path):
    print("Attempting to extract audio stream from video...")
    temp_audio_path = "temp_audio.wav"
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio_path]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(temp_audio_path):
        print("Audio extracted successfully."); return temp_audio_path
    else:
        print("[INFO] Could not extract audio. The video may be silent."); return None

def preprocess_video(video_path, feature_extractor):
    IMG_SIZE = 224; NUM_FRAMES = 16
    # ... (rest of function is unchanged)
    if not os.path.exists(video_path): print(f"[ERROR] Video file not found: {video_path}"); return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"[ERROR] Could not open video: {video_path}"); return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ret, frame = cap.read()
        if ret: frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)))
    cap.release()
    if not frames: return None
    frames_np = np.array(frames); preprocessed_frames = tf.keras.applications.resnet50.preprocess_input(frames_np)
    frame_features = feature_extractor.predict(preprocessed_frames, verbose=0)
    return np.mean(frame_features, axis=0)
    
def preprocess_audio(audio_path):
    SAMPLE_RATE = 16000; N_MFCC = 40; MAX_LEN = 174
    # ... (rest of function is unchanged)
    if not os.path.exists(audio_path): print(f"[ERROR] Audio file not found: {audio_path}"); return None
    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfccs = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    if mfccs.shape[1] > MAX_LEN: mfccs = mfccs[:, :MAX_LEN]
    else: mfccs = np.pad(mfccs, ((0, 0), (0, MAX_LEN - mfccs.shape[1])), mode='constant')
    return mfccs

def main():
    """Main function with robust logic to handle silent videos and send Discord alerts."""
    print("--- Live Violence Detection System (with Discord) ---")
    
    video_model = tf.keras.models.load_model(VIDEO_MODEL_PATH)
    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
    video_feature_extractor = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    temp_audio_path = None
    try:
        # VIDEO ANALYSIS
        print(f"\nProcessing video stream from: {os.path.basename(INPUT_VIDEO_PATH)}...")
        video_features = preprocess_video(INPUT_VIDEO_PATH, video_feature_extractor)
        if video_features is None: sys.exit("[EXIT] CRITICAL: Could not process video stream.")
        video_prediction = video_model.predict(np.expand_dims(video_features, axis=0))[0][0]

        # AUDIO ANALYSIS
        temp_audio_path = extract_audio(INPUT_VIDEO_PATH)
        if temp_audio_path:
            print(f"Processing audio stream...")
            audio_features = preprocess_audio(temp_audio_path)
            audio_prediction = audio_model.predict(np.transpose(np.expand_dims(audio_features, axis=0), (0, 2, 1)))[0][0]
            final_prediction = (video_prediction + audio_prediction) / 2
            analysis_type = "Ensemble (Audio+Video)"
        else:
            final_prediction = video_prediction
            audio_prediction = None
            analysis_type = "Video-Only"

        # FINAL VERDICT
        print("\n--- DETECTION RESULTS ---")
        print(f"Analysis Type: {analysis_type}")
        print(f"Video Model Confidence: {video_prediction * 100:.2f}%")
        if audio_prediction is not None: print(f"Audio Model Confidence: {audio_prediction * 100:.2f}%")
        print(f"\nFinal Combined Confidence Score: {final_prediction * 100:.2f}%")

        if final_prediction > 0.5:
            final_verdict = "VIOLENCE DETECTED"
            print(f"Final Verdict: {final_verdict}. Sending notification...")
            send_discord_notification(final_verdict, final_prediction)
        else:
            final_verdict = "Non-Violence Detected"
            print(f"Final Verdict: {final_verdict}.")

    finally:
        # Cleanup
        if temp_audio_path and os.path.exists(temp_audio_path):
            print("Cleaning up temporary audio file...")
            os.remove(temp_audio_path)

if __name__ == '__main__':
    main()