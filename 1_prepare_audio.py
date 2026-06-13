import os
import random
import shutil
from moviepy import VideoFileClip

# --- Configuration ---
# Source paths based on your directory structure
VIDEO_ROOT = r"C:\Users\sanab\OneDrive\Desktop\Amakh+1\content\Real Life Violence Dataset"
AUDIO_SOURCE = r"C:\Users\sanab\OneDrive\Desktop\Amakh+1\data"
# Destination for extracted/paired audio
AUDIO_DEST = r"C:\Users\sanab\OneDrive\Desktop\Amakh+1\processed_audio"

ACTIONS = ['Violence', 'NonViolence']

def get_all_audio_files(source_dir):
    """Recursively finds all .wav files in the audio data directory."""
    audio_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def prepare_audio():
    print("Starting audio preparation and pairing...")
    
    # Create destination folders
    for action in ACTIONS:
        os.makedirs(os.path.join(AUDIO_DEST, action), exist_ok=True)

    # Get list of all available audio samples
    available_audio = get_all_audio_files(AUDIO_SOURCE)
    
    if not available_audio:
        print("Error: No audio files found in source directory!")
        return

    for action in ACTIONS:
        video_folder = os.path.join(VIDEO_ROOT, action)
        audio_save_folder = os.path.join(AUDIO_DEST, action)
        
        videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
        print(f"Processing {len(videos)} videos for category: {action}")

        for v_file in videos:
            # Generate a corresponding audio filename
            video_name = os.path.splitext(v_file)[0]
            target_audio_path = os.path.join(audio_save_folder, f"{video_name}.wav")

            # Strategy: Randomly pick an audio file from your dataset to pair with the video
            # This creates the "Multi-modal" training set
            chosen_audio = random.choice(available_audio)
            
            try:
                # Copy the audio file and rename it to match the video
                shutil.copy(chosen_audio, target_audio_path)
            except Exception as e:
                print(f"Failed to pair audio for {v_file}: {e}")

    print(f"\nSuccess! Paired audio files are saved in: {AUDIO_DEST}")

if __name__ == "__main__":
    prepare_audio()