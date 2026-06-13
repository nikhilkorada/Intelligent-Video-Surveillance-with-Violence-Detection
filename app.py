import os
import cv2
import numpy as np
import librosa
import collections
import requests
import time
import threading
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 1. CONFIGURATION ---
# Replace with your actual Discord Webhook URL
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1478280914009194557/HN4K1hvHO2keVHnvWHuZgKfCG3K7P_fU6DaquZHIgz45GC2KtEZuV8nRLr8ubw_wnHpz"
ALERT_COOLDOWN = 20
last_alert_time = 0

# --- 2. LOAD MODELS ---
print("\n[1/4] Loading Multi-modal LSTM Model...")
model = load_model('violence_model.h5')

print("[2/4] Loading YOLOv8-Pose...")
pose_model = YOLO('yolov8n-pose.pt')

print("[3/4] Loading Standalone Audio Model...")

AUDIO_ONLY_MODEL_PATH = 'audio_violence_classifier.h5' 
audio_only_model = load_model(AUDIO_ONLY_MODEL_PATH)

# Global state for live feed
video_buffer = collections.deque(maxlen=30)
current_status = {"label": "Normal", "prob": 0.0}

# --- 3. HELPER FUNCTIONS ---
def get_location_data():
    """Fetches approximate location and coordinates with multiple fallbacks."""
    # Attempt 1: ip-api (No SSL required, very reliable for local)
    try:
        response = requests.get('http://ip-api.com/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                city = data.get('city', 'Unknown City')
                region = data.get('regionName', 'Unknown Region')
                country = data.get('country', 'Unknown Country')
                lat, lon = data.get('lat'), data.get('lon')
                return f"{city}, {region}, {country}", f"https://www.google.com/maps?q={lat},{lon}"
    except Exception:
        pass

    # Attempt 2: ipapi.co (Fallback)
    try:
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            city = data.get('city', 'Unknown City')
            lat, lon = data.get('latitude'), data.get('longitude')
            return f"{city}, {data.get('region')}, {data.get('country_name')}", f"https://www.google.com/maps?q={lat},{lon}"
    except Exception:
        pass

    return "Location Detection Failed", None

# Cache location at startup
GLOBAL_LOCATION_TEXT, GLOBAL_MAPS_URL = get_location_data()
print(f"[INFO] System Location: {GLOBAL_LOCATION_TEXT}")

def send_discord_alert(frame, probability, mode="Video"):
    """Sends notification with clickable map link and image to Discord."""
    global last_alert_time
    try:
        location_display = f"[📍 {GLOBAL_LOCATION_TEXT}]({GLOBAL_MAPS_URL})" if GLOBAL_MAPS_URL else f"📍 `{GLOBAL_LOCATION_TEXT}`"

        payload = {
            "content": (
                f"🚨 **AI SECURITY ALERT ({mode})** 🚨\n"
                f"**Status:** ⚔️ Violence Detected\n"
                f"**Confidence:** `{probability}%` \n"
                f"**Location:** {location_display}\n"
                f"**Timestamp:** `{time.ctime()}`"
            )
        }
        
        if frame is not None:
            _, img_encoded = cv2.imencode('.jpg', frame)
            # Use 'payload_json' to mix file and text data
            files = {
                "file": ("incident.jpg", img_encoded.tobytes(), "image/jpeg"),
                "payload_json": (None, json.dumps(payload)) 
            }
            response = requests.post(DISCORD_WEBHOOK_URL, files=files, timeout=10)
        else:
            response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
            
        print(f"[DEBUG] Discord Response: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Discord sync failed: {e}")

# --- 4. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/audio_detect')
def audio_detect_page():
    return render_template('audio_detect.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            y, sr = librosa.load(filepath, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            if mfcc.shape[1] > 174: mfcc = mfcc[:, :174]
            else: mfcc = np.pad(mfcc, ((0, 0), (0, 174 - mfcc.shape[1])), mode='constant')
            
            input_data = np.transpose(np.expand_dims(mfcc, axis=0), (0, 2, 1))
            prediction = audio_only_model.predict(input_data, verbose=0)[0][0]
            prob = round(float(prediction) * 100, 2)
            
            if prediction > 0.80:
                threading.Thread(target=send_discord_alert, args=(None, prob, "Audio-Only")).start()

            os.remove(filepath)
            return jsonify({'verdict': "VIOLENCE DETECTED" if prediction > 0.5 else "Normal", 'confidence': f"{prob}%", 'is_violent': bool(prediction > 0.5)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            cap = cv2.VideoCapture(filepath)
            v_feats = []
            while len(v_feats) < 30:
                ret, frame = cap.read()
                if not ret: break
                res = pose_model(frame, verbose=False)
                if len(res[0].keypoints.data) > 0:
                    k = res[0].keypoints.data[0].cpu().numpy().flatten()
                    if len(k) == 51: v_feats.append(k)
            cap.release()

            if len(v_feats) < 30:
                if os.path.exists(filepath): os.remove(filepath)
                return render_template('upload.html', result="Error: Need 30 pose frames", prob=0)

            try:
                y, sr = librosa.load(filepath, sr=16000)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                a_feats = cv2.resize(mfcc, (30, 40)).T
                
                input_tensor = np.expand_dims(np.concatenate([np.array(v_feats), a_feats], axis=1), axis=0)
                pred = model.predict(input_tensor, verbose=0)
                prob = round(float(pred[0][0]) * 100, 2)

                if prob > 85:
                    threading.Thread(
                    target=send_discord_alert,
                    args=(None, prob, "Fusion Upload")
                    ).start()
                
                os.remove(filepath)
                return render_template('upload.html', result="Violence" if prob > 50 else "Normal", prob=prob)
            except Exception as e:
                if os.path.exists(filepath): os.remove(filepath)
                return render_template('upload.html', result="Fusion Error", prob=0)
    return render_template('upload.html')

# --- 5. LIVE FEED GENERATOR ---
def gen_frames():
    global current_status, last_alert_time
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        raw_frame = frame.copy()
        results = pose_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        if len(results[0].keypoints.data) > 0:
            kpts = results[0].keypoints.data[0].cpu().numpy().flatten()
            if len(kpts) == 51: video_buffer.append(kpts)

        if len(video_buffer) == 30:
            v_seq = np.array(video_buffer)
            # Fusing with zeros for audio during live demo for stability
            fused = np.expand_dims(np.concatenate([v_seq, np.zeros((30, 40))], axis=1), axis=0)
            prediction = model.predict(fused, verbose=0)
            prob = float(prediction[0][0])
            current_status["prob"] = round(prob * 100, 2)
            current_status["label"] = "Violence" if prob > 0.60 else "Normal"

            if current_status["label"] == "Violence" and prob > 0.85:
                if (time.time() - last_alert_time) > ALERT_COOLDOWN:
                    last_alert_time = time.time()
                    threading.Thread(target=send_discord_alert, args=(raw_frame, current_status["prob"])).start()

        if current_status["label"] == "Violence":
            cv2.rectangle(annotated_frame, (0,0), (640, 50), (0, 0, 255), -1)
            cv2.putText(annotated_frame, f"VIOLENCE: {current_status['prob']}%", (150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

if __name__ == '__main__':
    print("[4/4] --- SERVER STARTING ON http://127.0.0.1:5000 ---")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)