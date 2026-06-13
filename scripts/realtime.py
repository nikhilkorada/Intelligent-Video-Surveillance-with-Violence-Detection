import os
import sys
import threading
import subprocess
import numpy as np
import tensorflow as tf
import librosa
import cv2
import requests # For sending Discord notifications
from dotenv import load_dotenv
import queue # For thread-safe data sharing
import time
import sounddevice as sd # For real-time audio capture

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# --- 1. CONFIGURATION ---
# Use 0 for default webcam, or a different index if you have multiple cameras
INPUT_SOURCE = 0

# Load Discord Webhook URL securely from the .env file
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

VIDEO_MODEL_PATH = '../output/trained_model/video_violence_classifier.h5'
AUDIO_MODEL_PATH = '../output/trained_model/audio_violence_classifier.h5'

# Model and preprocessing constants
IMG_SIZE = 224
NUM_FRAMES = 16 # Number of frames for video sequence
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 174 # Max length for MFCC features

# --- 2. NOTIFICATION FUNCTION (DISCORD) ---
def send_discord_notification(final_verdict, confidence_score):
    """Sends a notification to a Discord channel using a Webhook."""
    if not DISCORD_WEBHOOK_URL:
        print("\n[WARNING] DISCORD_WEBHOOK_URL not found in .env file. Skipping notification.")
        return
    
    message = f"🚨 **ALERT: {final_verdict}** 🚨\n> **Confidence Score:** `{confidence_score * 100:.2f}%`"
    data = {"content": message, "username": "Violence Detection Bot"}
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        response.raise_for_status()
        print("\nDiscord notification sent successfully!")
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Failed to send Discord notification: {e}")

# --- 3. MULTITHREADED PROCESSING ---

class VideoStreamProcessor(threading.Thread):
    """
    Thread to capture frames, preprocess them, and put them in a queue for a separate thread to process.
    """
    def __init__(self, frame_queue, frames_buffer, feature_extractor, video_model):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(INPUT_SOURCE)
        if not self.cap.isOpened():
            print("[ERROR] Could not open video source.")
            sys.exit(1)
        self.frame_queue = frame_queue
        self.frames_buffer = frames_buffer
        self.feature_extractor = feature_extractor
        self.video_model = video_model
        self.daemon = True # Thread will close when the main program exits
        self.running = True
        
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[INFO] No more frames from stream. Exiting.")
                self.running = False
                break
            
            # Preprocess the frame for the model
            processed_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
            
            # Add the new frame to the buffer
            self.frames_buffer.append(processed_frame)
            
            # If buffer is full, process a chunk
            if len(self.frames_buffer) >= NUM_FRAMES:
                # Take the last NUM_FRAMES frames from the buffer
                frames_to_process = np.array(list(self.frames_buffer)[-NUM_FRAMES:])
                
                # Preprocess for ResNet50
                preprocessed_frames = tf.keras.applications.resnet50.preprocess_input(frames_to_process)
                
                # Get features and add to queue
                frame_features = self.feature_extractor.predict(preprocessed_frames, verbose=0)
                video_features = np.mean(frame_features, axis=0)
                self.frame_queue.put(video_features)
                
            # Display the live feed
            cv2.imshow('Live Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Video stream thread stopped.")

class AudioStreamProcessor(threading.Thread):
    """
    Thread to capture audio, preprocess it, and put it in a queue.
    """
    def __init__(self, audio_queue, audio_model):
        threading.Thread.__init__(self)
        self.audio_queue = audio_queue
        self.audio_model = audio_model
        self.daemon = True
        self.running = True
        
    def run(self):
        print("Starting audio capture...")
        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            
            # Preprocess the audio chunk
            mfccs = librosa.feature.mfcc(y=indata.flatten(), sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            if mfccs.shape[1] > MAX_LEN:
                mfccs = mfccs[:, :MAX_LEN]
            else:
                mfccs = np.pad(mfccs, ((0, 0), (0, MAX_LEN - mfccs.shape[1])), mode='constant')
            
            # Add to queue for inference
            self.audio_queue.put(mfccs)
        
        try:
            # Use sounddevice to capture audio. The blocksize determines chunk size.
            with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=SAMPLE_RATE * 2):
                while self.running:
                    time.sleep(0.1) # Keep the thread alive
        except Exception as e:
            print(f"[ERROR] Audio capture failed: {e}")
            self.running = False
        
        print("Audio stream thread stopped.")

def main():
    """Main function with real-time, multithreaded logic."""
    print("--- Real-Time Violence Detection System (with Discord) ---")
    
    # Load models
    try:
        video_model = tf.keras.models.load_model(VIDEO_MODEL_PATH)
        audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
        video_feature_extractor = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    except Exception as e:
        print(f"[EXIT] Could not load models. Check paths. Error: {e}")
        return

    # Queues for inter-thread communication
    video_features_queue = queue.Queue()
    audio_features_queue = queue.Queue()
    
    # Buffer to hold recent video frames
    video_frames_buffer = []

    # Start the video and audio processing threads
    video_thread = VideoStreamProcessor(video_features_queue, video_frames_buffer, video_feature_extractor, video_model)
    audio_thread = AudioStreamProcessor(audio_features_queue, audio_model)
    
    video_thread.start()
    audio_thread.start()
    
    last_audio_prediction = None
    
    try:
        while video_thread.is_alive() or audio_thread.is_alive():
            
            # --- VIDEO INFERENCE ---
            if not video_features_queue.empty():
                video_features = video_features_queue.get()
                video_prediction = video_model.predict(np.expand_dims(video_features, axis=0))[0][0]
                
                # --- AUDIO INFERENCE ---
                # Grab the latest audio prediction from the queue if available
                if not audio_features_queue.empty():
                    audio_features = audio_features_queue.get()
                    last_audio_prediction = audio_model.predict(np.transpose(np.expand_dims(audio_features, axis=0), (0, 2, 1)))[0][0]

                # --- COMBINED VERDICT ---
                if last_audio_prediction is not None:
                    final_prediction = (video_prediction + last_audio_prediction) / 2
                    analysis_type = "Ensemble (Audio+Video)"
                else:
                    final_prediction = video_prediction
                    analysis_type = "Video-Only"
                
                # Display and act on results
                print("\n--- DETECTION RESULTS ---")
                print(f"Analysis Type: {analysis_type}")
                print(f"Video Confidence: {video_prediction * 100:.2f}%")
                if last_audio_prediction is not None:
                    print(f"Audio Confidence: {last_audio_prediction * 100:.2f}%")
                print(f"Final Combined Confidence: {final_prediction * 100:.2f}%")
                
                if final_prediction > 0.5:
                    final_verdict = "VIOLENCE DETECTED"
                    print(f"Final Verdict: {final_verdict}. Sending notification...")
                    send_discord_notification(final_verdict, final_prediction)
                else:
                    final_verdict = "Non-Violence Detected"
                    print(f"Final Verdict: {final_verdict}.")
            
            # Avoid a tight loop
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping threads...")
    finally:
        video_thread.running = False
        audio_thread.running = False
        video_thread.join()
        audio_thread.join()
        print("All threads stopped. Exiting.")


if __name__ == '__main__':
    main()