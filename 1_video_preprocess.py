import cv2
import os

# --- Configuration ---
# Source: Your raw dataset path
# Destination: A folder for cleaned, standardized videos
INPUT_ROOT = r"C:\Users\sanab\OneDrive\Desktop\Amakh+1\content\Real Life Violence Dataset"
OUTPUT_ROOT = r"C:\Users\sanab\OneDrive\Desktop\Amakh+1\preprocessed_videos"

ACTIONS = ['Violence', 'NonViolence']
TARGET_SIZE = (640, 480) # YOLOv8 works best with these dimensions
TARGET_FPS = 30          # Standardize frame rate for LSTM temporal consistency

def preprocess_videos():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    for action in ACTIONS:
        input_dir = os.path.join(INPUT_ROOT, action)
        output_dir = os.path.join(OUTPUT_ROOT, action)
        os.makedirs(output_dir, exist_ok=True)

        video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi'))]
        print(f"\nStandardizing {len(video_files)} videos for category: {action}")

        for v_file in video_files:
            input_path = os.path.join(input_dir, v_file)
            output_path = os.path.join(output_dir, v_file)

            cap = cv2.VideoCapture(input_path)
            
            # Define codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, TARGET_SIZE)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Resize frame
                resized_frame = cv2.resize(frame, TARGET_SIZE)

                # 2. Basic Image Enhancement (Optional: helps YOLO in dark scenes)
                # You could add brightness/contrast normalization here if needed

                out.write(resized_frame)

            cap.release()
            out.release()
            print(f"  [Done] {v_file}")

    print(f"\nPreprocessing Complete. Saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    preprocess_videos()