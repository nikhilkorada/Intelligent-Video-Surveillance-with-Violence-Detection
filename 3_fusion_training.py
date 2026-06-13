import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.utils import to_categorical

# Paths
VIDEO_BASE = 'extracted_features/video'
AUDIO_BASE = 'extracted_features/audio'
ACTIONS = ['Violence', 'NonViolence']

def create_fused_dataset():
    X, y = [], []
    
    for label, action in enumerate(ACTIONS):
        v_path = os.path.join(VIDEO_BASE, action)
        a_path = os.path.join(AUDIO_BASE, action)
        
        v_files = [os.path.join(v_path, f) for f in os.listdir(v_path) if f.endswith('.npy')]
        a_files = [os.path.join(a_path, f) for f in os.listdir(a_path) if f.endswith('.npy')]
        
        print(f"Action: {action} | Videos: {len(v_files)} | Audio: {len(a_files)}")
        
        # We create pairs. We'll use the count of the larger set to ensure we use all data
        num_samples = max(len(v_files), len(a_files))
        
        for i in range(num_samples):
            # Randomly pair video and audio from the same class
            v_data = np.load(random.choice(v_files)) # Shape (30, 51)
            a_data = np.load(random.choice(a_files)) # Shape (30, 40)
            
            # Feature Fusion: Concatenate (30, 51) + (30, 40) -> (30, 91)
            fused = np.concatenate([v_data, a_data], axis=1)
            X.append(fused)
            y.append(label)
            
    return np.array(X), to_categorical(y).astype(int)

# --- Build the Model ---
def build_model():
    model = Sequential([
        Input(shape=(30, 91)),
        LSTM(64, return_sequences=True, activation='tanh'),
        BatchNormalization(),
        LSTM(128, return_sequences=False, activation='tanh'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Execute ---
print("Fusing Features and Preparing Dataset...")
X, y = create_fused_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = build_model()
print("\nStarting Multi-Modal LSTM Training...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

model.save('violence_model.h5')
print("\nSuccess! 'violence_model.h5' created.")