import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.utils import to_categorical

# --- 1. Hardcoded Path Correction ---
# Since we know exactly where your files are now:
BASE_DIR = 'extracted_features'
ACTIONS = ['Violence', 'NonViolence']

def load_and_fuse_data():
    X, y = [], []
    
    for label, action in enumerate(ACTIONS):
        # Corrected paths to match your 'ls' output
        video_dir = os.path.join(BASE_DIR, 'video', action)
        audio_dir = os.path.join(BASE_DIR, 'audio', action)
        
        if not os.path.exists(video_dir):
            print(f"Directory not found: {video_dir}")
            continue
            
        # Get all .npy files in the video folder
        files = [f for f in os.listdir(video_dir) if f.endswith('.npy')]
        print(f"Found {len(files)} samples for: {action}")
        
        for file in files:
            v_path = os.path.join(video_dir, file)
            a_path = os.path.join(audio_dir, file)
            
            # Ensure the audio file exists for this video file
            if os.path.exists(a_path):
                v_data = np.load(v_path) # (30, 51)
                a_data = np.load(a_path) # (30, 40)
                
                # Late Fusion: Join Pose and Audio features
                fused_data = np.concatenate([v_data, a_data], axis=1) # (30, 91)
                X.append(fused_data)
                y.append(label)
            else:
                # If audio is missing, we skip to maintain data integrity
                continue

    return np.array(X), to_categorical(y, num_classes=2).astype(int)

# --- 2. Build the Multi-Modal LSTM ---
def build_model():
    model = Sequential([
        Input(shape=(30, 91)),
        LSTM(64, return_sequences=True, activation='tanh'),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(128, return_sequences=False, activation='tanh'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 3. Execution ---
if __name__ == "__main__":
    print("Initializing Data Fusion...")
    X, y = load_and_fuse_data()

    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Data ready. Total samples: {len(X)}")

        
        
        model = build_model()
        print("\nStarting Training...")
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        model.save('violence_model.h5')
        print("\nSuccess! Model saved as 'violence_model.h5'")
    else:
        print("Fatal Error: Still no data found. Check if 'extracted_features/audio' folder exists.")