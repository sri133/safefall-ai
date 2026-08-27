# 🏥 SafeFall AI — Elderly Fall Detection & Monitoring System

**CareVision HealthTech Pvt. Ltd.** | AI-Powered Computer Vision Healthcare Monitoring

SafeFall AI is a real-time elderly fall detection system built on pose estimation and deep learning. It detects human posture, classifies activity (Fall Detected, Walking, Sitting, Standing, Normal Activity), and fires emergency alerts the moment a fall occurs — deployed as a live, interactive Streamlit dashboard.

**🔗 Live App:** [safefall-ai.streamlit.app](https://safefall-ai-2pfgkwff6vvfgon4kd6scz.streamlit.app/)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Live App Pages](#live-app-pages)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation Results](#evaluation-results)
- [Project Structure](#project-structure)
- [Setup & Deployment](#setup--deployment)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)

---

## Overview

Elderly individuals living alone face serious risk during fall incidents — immediate medical help is often unavailable, and existing solutions (manual supervision, wearables, passive CCTV) are reactive, not proactive. SafeFall AI addresses this with an automated computer-vision pipeline:

**Video/Image → Pose Estimation → Feature Extraction → Deep Learning Classification → Alert Generation → Live Dashboard**

The system was built in two phases:
- **FA-1** — Problem definition, dataset preprocessing, and exploratory data analysis
- **FA-2** — Model training, evaluation, and full deployment

---

## Live App Pages

| Page | What it shows |
|---|---|
| **📹 Live Monitoring** (main) | Upload an image/video, get live pose overlay, activity prediction, alerts, and analytics |
| **📊 Model Performance** (tab) | Training accuracy/loss curves, confusion matrix, classification report |
| **📊 EDA Dashboard** (sidebar page) | Dataset composition — bar chart, pie chart, activity distribution, summary table |
| **🔬 Model Insights** (sidebar page) | Precision/Recall/F1 radar chart, NN vs RF trade-off line chart, ranked metrics funnel chart |

---

## Features

1. **Live Pose Visualization** — skeleton overlay drawn directly on the uploaded frame with a color-coded prediction banner
2. **Joint-Angle Radar Chart** — live 5-axis posture "fingerprint" (torso tilt, knee, hip, elbow, shoulder angles)
3. **Fall Probability Timeline** — per-frame fall-probability line chart across an analyzed video, NN vs RF overlaid
4. **Model Comparison** — Dense NN and Random Forest predictions shown side-by-side on the same input
5. **Alert History Log** — every triggered alert logged with frame number, confidence, and thumbnail
6. **Audible Sound Alert** — an in-code-generated beep tone plays automatically when a fall fires
7. **Processing Speed Meter** — live ms/frame and implied FPS, demonstrating real-time viability
8. **Downloadable PDF Incident Report** — one-click report with summary stats, distribution chart, and alert log
9. **Shift Report Card** — end-of-session summary styled like a hospital shift-change report
10. **Dark "Control Room" Theme** — custom CSS + Plotly dark styling throughout, glowing pulse animation on alerts

---

## Dataset

**Source:** [Le2i Fall Dataset (IMVIA Laboratory, University of Burgundy)](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia) — real-world surveillance footage across 6 indoor scenes (Coffee_room_01/02, Home_01/02, Lecture_room, Office).

| Metric | Value |
|---|---|
| Videos processed | 190 |
| Total frames extracted (sample_every=5) | 8,366 |
| Fall frames | 456 (5.5%) |
| Not-fall frames | 7,910 (94.5%) |
| Frames with usable pose detected | 6,311 (75.4%) |
| Frames dropped (no pose detected) | 2,055 (24.6%) |
| Final labeled set — fall | 342 |
| Final labeled set — not_fall | 5,969 |

Annotations mark only the **fall event window** (start/end frame) per video — not per-frame labels for walking/sitting/standing. This shaped the modeling approach below.

---

## Model Architecture

**Pose Estimation:** MediaPipe Pose (33 landmarks → normalized 99-feature vector: x, y, visibility per key joint, scale/position-invariant relative to torso).

**Classification — a two-stage hybrid approach:**

1. **Trained Dense Neural Network** (128 → Dropout → 64 → Dropout → softmax) classifies each frame as **Fall / Not-Fall**, using real annotated ground truth. Trained with class weighting to counter the ~17:1 imbalance. A Random Forest (300 trees) was trained in parallel as a comparison baseline.
2. **Geometric rule layer** further splits Not-Fall frames into **Walking / Sitting / Standing / Normal Activity** using joint-angle heuristics (knee bend, torso tilt, hip-to-ankle distance, frame-to-frame ankle motion) — since the dataset doesn't provide ground truth for these sub-classes.
3. **Safety-net override:** a near-horizontal torso angle triggers "Fall Detected" regardless of the NN's raw prediction, catching cases where someone remains lying down after a fall (a gap in the dataset's fall-window-only labeling).

**Data split:** 70% train / 15% validation / 15% test, stratified.

---

## Evaluation Results

Test set: 947 frames (52 fall / 895 not_fall).

| Model | Accuracy | Fall Precision | Fall Recall | Fall F1 |
|---|---|---|---|---|
| **Dense NN** (deployed) | 88.9% | 27.0% | **59.6%** | 37.1% |
| Random Forest | **95.1%** | 75.0% | 17.3% | 28.1% |

**Key finding:** Random Forest wins on raw accuracy, but misses over 4 in 5 real falls. For a safety-critical system, recall on the fall class matters more than overall accuracy — this is why the deployed app uses the Dense NN despite its lower accuracy score.

Full breakdowns (confusion matrix, training curves, ROC-style comparisons) are rendered live in the app's **Model Performance** tab and **Model Insights** page.

---

## Project Structure

```
fall_detection_project/
├── app.py                        # Main Streamlit dashboard
├── utils.py                      # Shared pose/feature/heuristic functions
├── pages/
│   ├── 1_EDA_Dashboard.py        # FA-1 EDA charts (bar, pie, table, distribution)
│   └── 2_Model_Insights.py       # FA-2 evaluation charts (radar, line, funnel)
├── 1_extract_frames_and_labels.py # Step 4/5: video → labeled frames
├── 2_extract_pose_landmarks.py    # Step 4/5: frames → pose feature CSV
├── 3_train_model.py               # Step 5: trains Dense NN + Random Forest
├── 4_evaluate_model.py            # Step 6: metrics + confusion matrix
├── model/                         # Trained artifacts (committed to repo)
│   ├── model_nn.keras
│   ├── model_rf.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   ├── test_split.csv
│   ├── training_history.png
│   ├── confusion_matrix_nn.png
│   └── nn_classification_report.txt
├── requirements.txt               # Python dependencies
├── packages.txt                   # System-level apt dependencies (OpenCV support)
└── .streamlit/
    └── config.toml                # Upload size limit config
```

---

## Setup & Deployment

### Run the training pipeline (Google Colab)
```python
# 1. Download dataset from Kaggle
!kaggle datasets download -d tuyenldvn/falldataset-imvia -p /content
!unzip -q /content/falldataset-imvia.zip -d /content/falldataset-imvia

# 2. Extract & label frames
!python 1_extract_frames_and_labels.py --dataset_root /content/falldataset-imvia --out_dir /content/frames --sample_every 5

# 3. Extract pose landmarks
!python 2_extract_pose_landmarks.py --labels_csv /content/frames/labels.csv --out_csv /content/landmarks.csv

# 4. Train models
!python 3_train_model.py --landmarks_csv /content/landmarks.csv --out_dir /content/model --epochs 60

# 5. Evaluate
!python 4_evaluate_model.py --model_dir /content/model
```

### Deploy the app (Streamlit Cloud)
1. Push this repository to GitHub, including the `model/` folder with trained artifacts.
2. Go to [share.streamlit.io](https://streamlit.io/cloud) → **New app** → point at this repo, main file `app.py`.
3. Streamlit Cloud auto-installs `requirements.txt` and `packages.txt`.
4. Deploy — the app is live at the generated `.streamlit.app` URL.

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Known Limitations

- **Severe class imbalance** (~5.5% fall frames) — addressed via class weighting, but still limits fall-class precision.
- **Pose detection fails on ~24.6% of frames** — driven by occlusion, unusual camera angles, and partial visibility, especially in Office/Lecture_room scenes.
- **No true multi-class ground truth** — Walking/Sitting/Standing/Normal rely on geometric heuristics, not trained labels, since the dataset only annotates fall timing.
- **Single-frame classification** — no temporal modeling (e.g., LSTM), so the system reacts to static pose rather than fall *motion*.
- **Domain shift risk** — the model is trained on wide-angle CCTV-style footage; performance on close-up webcam angles (e.g., a laptop camera) is expected to degrade since key joints (knees, ankles) are often out of frame.
- **Streamlit Cloud free-tier memory limits** — very large video uploads (400MB+) can cause the app to crash; recommend clips under ~100MB for reliable analysis.

---

## Future Improvements

- Collect or manually annotate per-frame activity labels to replace the geometric sub-classifier with a trained one
- Explore temporal models (CNN+LSTM) to capture fall *motion*, not just end pose
- Require multiple consecutive fall-frames before alerting, to reduce false alarms
- Add low-light/brightness augmentation for better night-time performance
- Support live CCTV/webcam feeds for real-time deployment
- Periodic retraining as new annotated healthcare data becomes available

---

## Tech Stack

- **Pose Estimation:** MediaPipe Pose
- **Deep Learning:** TensorFlow / Keras
- **Classical ML baseline:** Scikit-learn (Random Forest)
- **Computer Vision:** OpenCV
- **Dashboard:** Streamlit, Plotly, Matplotlib
- **Deployment:** Streamlit Community Cloud
- **Data Processing:** Pandas, NumPy

---

## Course Context

Built for **Machine Learning and Deep Learning — Formative Assessment 1 & 2**, simulating CareVision HealthTech Pvt. Ltd.'s SafeFall AI initiative for elderly wellness and smart healthcare monitoring.
