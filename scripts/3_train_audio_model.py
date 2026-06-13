import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. CONFIGURATION ---
PROCESSED_DATA_DIR = '../output/processed_data/'
MODEL_OUTPUT_DIR = '../output/trained_model/'

# --- 2. LOAD THE PROCESSED DATA ---
print("--- Building Audio Expert (Step 3): Training the Model ---")
print("Loading pre-processed audio feature data...")

try:
    X = np.load(os.path.join(PROCESSED_DATA_DIR, 'audio_features.npy'))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, 'audio_labels.npy'))
except FileNotFoundError:
    print(f"\n[ERROR] Processed data files not found in '{PROCESSED_DATA_DIR}'. Please run the feature extraction script first.")
    exit()

print(f"Features loaded with shape: {X.shape}")
print(f"Labels loaded with shape: {y.shape}")

# --- 3. SPLIT DATA INTO TRAINING AND TESTING SETS ---
# Stratify=y ensures both sets have a similar balance of violent/non-violent examples.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Reshape data for the 1D CNN: (num_samples, timesteps, features)
# Here, MFCCs are features and the length is the timestep.
X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

print(f"Reshaped training data for CNN: {X_train.shape}")

# --- 4. BUILD THE 1D CNN MODEL ---
print("\nBuilding the 1D Convolutional Neural Network model...")

model = Sequential([
    # Input layer: Shape is (timesteps, features)
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(), # Flatten the output to feed into the dense layers
    
    Dense(256, activation='relu'),
    Dropout(0.5),

    # Output layer: 1 neuron with a sigmoid activation for binary (0 or 1) classification
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 5. TRAIN THE MODEL ---
print("\nTraining the model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    epochs=50, # Train for more epochs; early stopping will handle overfitting
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# --- 6. EVALUATE THE MODEL ---
print("\nEvaluating the final model on the test set...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# --- 7. SAVE THE TRAINED MODEL ---
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
model_save_path = os.path.join(MODEL_OUTPUT_DIR, 'audio_violence_classifier.h5')
model.save(model_save_path)
print(f"\nModel saved successfully to: {model_save_path}")

print("\n--- Audio Expert Model Training Complete! ---")