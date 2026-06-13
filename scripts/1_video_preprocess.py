import os
import pandas as pd

# --- 1. CONFIGURATION ---
# Paths to your data folders. Please verify they are correct.
NON_VIOLENCE_DIR = r'C:\Users\sanab\OneDrive\Desktop\Amakh+1\content\Real Life Violence Dataset\NonViolence'
VIOLENCE_DIR = r'C:\Users\sanab\OneDrive\Desktop\Amakh+1\content\Real Life Violence Dataset\Violence'

OUTPUT_CSV_FILE = '../data/video_metadata.csv'
# ---------------------------------------------------------------------------

def create_metadata():
    """
    Scans the Violence and NonViolence folders, creates a DataFrame of
    video paths and labels, shuffles it, and saves it to a CSV file.
    """
    print("--- Building Video Expert (Step 1): Preparing Video Metadata ---")
    
    file_list = []

    # --- Scan for Non-Violent videos (Label = 0) ---
    print(f"Scanning {NON_VIOLENCE_DIR}...")
    for filename in os.listdir(NON_VIOLENCE_DIR):
        if filename.endswith(('.mp4', '.avi', '.mov', '.wmv')):
            file_list.append({
                'video_path': os.path.join(NON_VIOLENCE_DIR, filename),
                'label': 0  # 0 for NonViolence
            })

    # --- Scan for Violent videos (Label = 1) ---
    print(f"Scanning {VIOLENCE_DIR}...")
    for filename in os.listdir(VIOLENCE_DIR):
        if filename.endswith(('.mp4', '.avi', '.mov', '.wmv')):
            file_list.append({
                'video_path': os.path.join(VIOLENCE_DIR, filename),
                'label': 1  # 1 for Violence
            })
            
    if not file_list:
        print("[ERROR] No video files found. Please check the paths in the configuration.")
        return
        
    # --- Create and Shuffle DataFrame ---
    metadata_df = pd.DataFrame(file_list)
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    
    # --- Save to CSV ---
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
    metadata_df.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print(f"\nSuccessfully found and labeled {len(metadata_df)} videos.")
    print(f"Metadata saved to: {OUTPUT_CSV_FILE}")
    print("\nSample of the metadata:")
    print(metadata_df.head())
    print("\n--- Video Data Preparation Complete ---")
    
if __name__ == '__main__':
    create_metadata()