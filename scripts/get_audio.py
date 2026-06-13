import os
import pandas as pd
from datasets import load_from_disk

# Configuration
LOCAL_DATASET_PATH = '../content/hemg_audio' 
OUTPUT_CSV_FILE = '../data/audio_metadata_fullpaths.csv'

def get_full_paths_surgically():
    """
    Extracts metadata by directly converting specific columns to a pandas DataFrame,
    bypassing the standard iteration that triggers the torchcodec error.
    """
    print("--- Step A (Surgical Approach): Getting full audio file paths ---")
    
    try:
        dataset_dict = load_from_disk(LOCAL_DATASET_PATH)
        dataset = dataset_dict['train']
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset from the local disk: {e}")
        return

    try:
        # --- THIS IS THE FIX ---
        # Instead of looping, we directly select the columns we need and convert
        # them to a pandas DataFrame. This is a lower-level operation that should
        # not trigger the full audio decoding process.
        print("Extracting metadata directly from the data table...")
        df = dataset.select_columns(["audio", "label"]).to_pandas()
        # ---------------------

        # Now that we have the data, extract the full path from the 'audio' column's dictionary
        df['full_audio_path'] = df['audio'].apply(lambda x: x['path'])
        
        # We no longer need the complex 'audio' column
        df = df.drop(columns=['audio'])

    except Exception as e:
        print(f"\n[ERROR] Failed to extract metadata even with the direct method.")
        print("This indicates a very deep issue with the dataset's compatibility with your environment.")
        print(f"Details: {e}")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print(f"\nSuccessfully saved {len(df)} full audio paths.")
    print(f"New metadata file created at: {OUTPUT_CSV_FILE}")
    print(df.head())

if __name__ == '__main__':
    get_full_paths_surgically()