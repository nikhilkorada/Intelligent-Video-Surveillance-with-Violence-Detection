# Intelligent Video Surveillance System with Violence Detection

## Overview

This project presents a **Multimodal Violence Detection System** for intelligent video surveillance. The system combines **visual** and **audio** information to accurately identify violent activities in real-time environments. Visual features are extracted using a **YOLO-based Human Pose Estimation** model, while audio features are extracted using **Mel-Frequency Cepstral Coefficients (MFCCs)**. The fused features are analyzed using a **Long Short-Term Memory (LSTM)** network to classify activities as **Violent** or **Non-Violent**.

---

## Features

* Real-time violence detection
* YOLO-based human pose estimation
* MFCC-based audio feature extraction
* Multimodal audio-visual feature fusion
* LSTM-based temporal sequence modeling
* Email and alarm-based alert system
* Robust performance under low-light and crowded environments

---

## Dataset

### Video Dataset

* **Real-Life Violence Dataset**

  * Violence
  * NonViolence

### Audio Dataset

* **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

  * Angry and Fearful emotions mapped to Violence
  * Remaining emotions mapped to NonViolence

---

## Project Pipeline

```text
Video Input ──► YOLO Pose Estimation ──► Pose Features
                                                   │
                                                   ▼
Audio Input ──► MFCC Extraction ───────► Audio Features
                                                   │
                                                   ▼
                                         Feature Fusion
                                                   │
                                                   ▼
                                              LSTM Model
                                                   │
                                                   ▼
                                Violence / Non-Violence Classification
                                                   │
                                                   ▼
                                        Alert Generation
```

---

## Technology Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Librosa
* Ultralytics YOLO
* Scikit-Learn
* Visual Studio Code

---

## Model Architecture

```text
Input (30 × 91)
      ↓
LSTM (64 Units)
      ↓
Batch Normalization
      ↓
LSTM (128 Units)
      ↓
Batch Normalization
      ↓
Dense (64, ReLU)
      ↓
Dropout (0.4)
      ↓
Dense (2, Softmax)
```

---

## Performance

| Metric           | Value  |
| ---------------- | ------ |
| Precision        | 94%    |
| Recall           | 92%    |
| F1-Score         | 93%    |
| Processing Speed | 30 FPS |

### Environmental Robustness

| Condition             | Accuracy |
| --------------------- | -------- |
| Low Lighting          | 88%      |
| Crowded Environment   | 91%      |
| Various Camera Angles | 90%      |

---

## Installation

```bash
git clone https://github.com/your-username/violence-detection-system.git

cd violence-detection-system

pip install -r requirements.txt
```

---

## Run the Project

### Video Preprocessing

```bash
python preprocess_videos.py
```

### Audio Feature Extraction

```bash
python extract_audio_features.py
```

### Video Feature Extraction

```bash
python extract_video_features.py
```

### Train Model

```bash
python train_model.py
```

---

## Future Enhancements

* Multi-camera surveillance support
* Transformer-based architectures
* Edge-device deployment
* Advanced attention mechanisms
* Larger and more diverse datasets

---

## Authors

* Nikhil Korada
* Kuruba Vijayenda Varma
* Adabala Nagesh Satya Vinay

Department of Data Science and Business Systems
SRM Institute of Science and Technology, Chennai, India

---

## License

This project is intended for academic and research purposes.
