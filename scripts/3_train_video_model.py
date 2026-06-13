import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. CONFIGURATION ---
PROCESSED_DATA_DIR = '../output/processed_data/'
MODEL_OUTPUT_DIR = '../output/trained_model/'

# --- 2. LOAD THE PROCESSED DATA ---
print("--- Building Video Expert (Step 3): Training the Model ---")
print("Loading pre-processed feature data...")

try:
    X = np.load(os.path.join(PROCESSED_DATA_DIR, 'video_features.npy'))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, 'video_labels.npy'))
except FileNotFoundError:
    print("\n[ERROR] Processed data files not found.")
    print(f"Please make sure 'video_features.npy' and 'video_labels.npy' exist in the '{PROCESSED_DATA_DIR}' directory.")
    exit()

print(f"Features loaded with shape: {X.shape}")
print(f"Labels loaded with shape: {y.shape}")

# --- 3. SPLIT DATA INTO TRAINING AND TESTING SETS ---
# We'll use 80% for training and 20% for testing.
# `stratify=y` ensures that both training and test sets have a proportional representation of violent and non-violent examples.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 4. BUILD THE NEURAL NETWORK MODEL ---
print("\nBuilding the neural network model...")

model = Sequential([
    # Input layer: The shape must match the number of features (2048 from ResNet50)
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5), # Dropout is a regularization technique to help prevent overfitting
    
    Dense(256, activation='relu'),
    Dropout(0.5),
    
    # Output layer: 1 neuron with a sigmoid activation for binary (0 or 1) classification
    Dense(1, activation='sigmoid')
])

# Compile the model, defining the optimizer, loss function, and metrics to track
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model's architecture
model.summary()

# --- 5. TRAIN THE MODEL ---
print("\nTraining the model...")

# EarlyStopping will stop training if the validation loss doesn't improve for 3 consecutive epochs.
# This prevents overfitting and saves time.
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# We train for a set number of epochs and use the test set as validation data
# to monitor performance on unseen data during training.
history = model.fit(X_train, y_train,
                    epochs=25, # Increased epochs, but early stopping will prevent overfitting
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
model_save_path = os.path.join(MODEL_OUTPUT_DIR, 'video_violence_classifier.h5')
model.save(model_save_path)
print(f"\nModel saved successfully to: {model_save_path}")

print("\n--- Video Expert Model Training Complete! ---")