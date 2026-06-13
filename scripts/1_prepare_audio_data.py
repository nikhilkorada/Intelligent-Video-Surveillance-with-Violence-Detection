import os
import pandas as pd

# --- 1. CONFIGURATION ---
# IMPORTANT: Make sure this path points to the folder where you extracted the
# RAVDESS audio files. It should contain subfolders like 'Actor_01', 'Actor_02', etc.
RAVDESS_DATA_DIR = r'C:\Users\sanab\OneDrive\Desktop\Amakh+1\data\audio_speech_actors_01-24'

OUTPUT_CSV_FILE = '../data/audio_metadata.csv'
# ---------------------------------------------------------------------------


def prepare_ravdess_metadata():
    """
    Scans the RAVDESS directory, parses emotion from filenames, maps them to
    a binary violence/non-violence label, and saves the result to a CSV.
    """
    print("--- Building Audio Expert (Step 1): Preparing RAVDESS Metadata ---")
    
    # This mapping is based on the official RAVDESS documentation.
    # Filename part 3: 01 = neutral, 02 = calm, 03 = happy, 04 = sad,
    # 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    emotion_map = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    }

    # Our mapping from emotion to the binary violence/non-violence label
    violence_map = {
        "angry": 1, "fearful": 1, "surprised": 1, # Class 1: Violence proxy
        "neutral": 0, "calm": 0, "happy": 0, "sad": 0, "disgust": 0 # Class 0: Non-Violence
    }
    
    file_list = []
    
    print(f"Scanning directory: {RAVDESS_DATA_DIR}...")
    # Walk through all the subdirectories (Actor_01, Actor_02, etc.)
    for root, dirs, files in os.walk(RAVDESS_DATA_DIR):
        for filename in files:
            if filename.endswith('.wav'):
                try:
                    # Filename parts are separated by hyphens
                    parts = filename.split('-')
                    emotion_code = parts[2]
                    
                    # Look up the emotion text and then the violence label
                    emotion_text = emotion_map[emotion_code]
                    label = violence_map[emotion_text]
                    
                    full_path = os.path.join(root, filename)
                    
                    file_list.append({
                        'full_audio_path': full_path,
                        'label': label,
                        'emotion': emotion_text # Keep the original emotion for reference
                    })
                except (IndexError, KeyError):
                    # Skip any files that don't match the expected naming convention
                    print(f"Skipping malformed filename: {filename}")

    if not file_list:
        print(f"[ERROR] No .wav files found in '{RAVDESS_DATA_DIR}'. Please check the path.")
        return

    # Create and shuffle the DataFrame
    metadata_df = pd.DataFrame(file_list)
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
    metadata_df.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print(f"\nSuccessfully found and labeled {len(metadata_df)} audio files.")
    print(f"Metadata saved to: {OUTPUT_CSV_FILE}")
    print("\nSample of the metadata:")
    print(metadata_df.head())
    print("\nClass distribution:")
    print(metadata_df['label'].value_counts())
    print("\n--- Audio Data Preparation Complete ---")

if __name__ == '__main__':
    prepare_ravdess_metadata()