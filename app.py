"""
app.py
 
Step 7: Streamlit deployment.
 
A healthcare-monitoring style dashboard that:
  - Accepts an image OR video upload
  - Runs MediaPipe pose estimation + the trained NN classifier
  - Sub-classifies non-fall frames into Walking / Sitting / Standing / Normal
  - Shows pose skeleton overlay, prediction, confidence
  - Fires a visible emergency alert when a fall is detected
  - Shows monitoring analytics: activity counts, distribution chart
 
Run locally:
    streamlit run app.py
 
Deploy: push this repo to GitHub, then deploy on https://streamlit.io/cloud
pointing at app.py. Make sure model_nn.keras, model_rf.joblib,
scaler.joblib, label_encoder.joblib (from 3_train_model.py's out_dir) are
committed to the repo (or otherwise reachable at MODEL_DIR below).
"""
 
import os
import tempfile
from collections import Counter
 
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
 
from utils import (
    get_pose_landmarks, landmarks_to_dict, extract_feature_vector,
    sub_classify_activity, draw_pose_and_label, ankle_midpoint,
    is_fall_alert, is_lying_down, FEATURE_COLUMNS,
)
import mediapipe as mp
 
MODEL_DIR = os.environ.get("MODEL_DIR", "model")
CLASS_DISPLAY = {"fall": "Fall Detected", "not_fall": "Not Fall"}
 
st.set_page_config(page_title="SafeFall AI — Elderly Monitoring", layout="wide")
 
 
@st.cache_resource
def load_artifacts():
    nn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_nn.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    return nn_model, scaler, le
 
 
def predict_frame(frame_bgr, pose_estimator, nn_model, scaler, le, prev_ankle=None):
    """Returns (label_str, confidence, annotated_frame, new_ankle_pos) or None."""
    results = get_pose_landmarks(frame_bgr, pose_estimator)
    if results is None:
        return None
 
    lm_dict = landmarks_to_dict(results)
    feats = extract_feature_vector(lm_dict)
    if feats is None:
        return None
 
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    probs = nn_model.predict(feats_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    raw_label = le.classes_[pred_idx]
 
    ankle_pos = ankle_midpoint(lm_dict)
    ankle_motion = None
    if prev_ankle is not None:
        ankle_motion = float(np.linalg.norm(ankle_pos - prev_ankle))
 
    if raw_label == "fall":
        final_label = "Fall Detected"
    elif is_lying_down(lm_dict):
        # Geometric override: trained model said not_fall, but the torso
        # is near-horizontal, meaning the person is on the ground. This
        # catches "still lying there after the fall" frames that the
        # dataset's fall-window-only labels don't cover (see utils.py).
        final_label = "Fall Detected"
        confidence = max(confidence, 0.75)
    else:
        final_label = sub_classify_activity(lm_dict, ankle_motion=ankle_motion)
 
    alert = is_fall_alert(final_label, confidence)
    annotated = draw_pose_and_label(frame_bgr, results, final_label, confidence, alert)
    return final_label, confidence, annotated, ankle_pos
 
 
# ---------------- UI ----------------
st.title("🏥 SafeFall AI — Elderly Fall Detection & Monitoring")
st.caption(
    "CareVision HealthTech — Computer Vision based elderly activity monitoring. "
    "Pose estimation: MediaPipe Pose. Classifier: Dense Neural Network on pose "
    "landmarks."
)
 
try:
    nn_model, scaler, le = load_artifacts()
except Exception as e:
    st.error(
        f"Could not load model artifacts from '{MODEL_DIR}'. Make sure you've run "
        f"3_train_model.py and copied model_nn.keras, scaler.joblib and "
        f"label_encoder.joblib into that folder.\n\nError: {e}"
    )
    st.stop()
 
tab_monitor, tab_performance = st.tabs(["📹 Live Monitoring", "📊 Model Performance"])
 
with tab_monitor:
    col_upload, col_stats = st.columns([2, 1])
 
    with col_upload:
        mode = st.radio("Input type", ["Image", "Video"], horizontal=True)
        uploaded = st.file_uploader(
            "Upload an image or video of the monitored area",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        )
 
    alert_placeholder = st.empty()
    results_area = st.container()
 
    counts = Counter()
    confidences = []
 
    if uploaded is not None:
        if mode == "Image":
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
 
            with mp.solutions.pose.Pose(static_image_mode=True) as pose_estimator:
                out = predict_frame(frame, pose_estimator, nn_model, scaler, le)
 
            if out is None:
                st.warning("No person detected in this image.")
            else:
                label, conf, annotated, _ = out
                counts[label] += 1
                confidences.append(conf)
 
                with results_area:
                    c1, c2 = st.columns(2)
                    c1.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption="Pose visualization + prediction",
                    )
                    with c2:
                        st.metric("Predicted Activity", label)
                        st.metric("Confidence", f"{conf*100:.1f}%")
 
                if is_fall_alert(label, conf):
                    alert_placeholder.error(
                        "🚨 EMERGENCY ALERT: Fall detected! Notify caregiver immediately."
                    )
 
        else:  # Video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
 
            sample_every = st.slider("Analyse every Nth frame", 1, 15, 5)
            run_btn = st.button("Run analysis")
 
            if run_btn:
                progress = st.progress(0, text="Analysing video...")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
 
                frame_gallery = []
                prev_ankle = None
                idx = 0
                fall_events = []
 
                with mp.solutions.pose.Pose(static_image_mode=True) as pose_estimator:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        idx += 1
                        if idx % sample_every != 0:
                            continue
 
                        out = predict_frame(
                            frame, pose_estimator, nn_model, scaler, le, prev_ankle
                        )
                        progress.progress(min(idx / total_frames, 1.0))
                        if out is None:
                            continue
                        label, conf, annotated, prev_ankle = out
                        counts[label] += 1
                        confidences.append(conf)
 
                        if is_fall_alert(label, conf):
                            fall_events.append(idx)
 
                        if len(frame_gallery) < 12:
                            frame_gallery.append((annotated, label, conf, idx))
 
                cap.release()
                progress.empty()
 
                if fall_events:
                    alert_placeholder.error(
                        f"🚨 EMERGENCY ALERT: Fall detected at frame(s) "
                        f"{fall_events[:5]}{'...' if len(fall_events) > 5 else ''}. "
                        f"Notify caregiver immediately."
                    )
                else:
                    alert_placeholder.success("No falls detected in this video.")
 
                with results_area:
                    st.subheader("Sampled frame predictions")
                    grid_cols = st.columns(4)
                    for i, (img, label, conf, frame_idx) in enumerate(frame_gallery):
                        with grid_cols[i % 4]:
                            st.image(
                                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                caption=f"frame {frame_idx}: {label} ({conf*100:.0f}%)",
                            )
 
    with col_stats:
        st.subheader("Monitoring Analytics")
        total = sum(counts.values())
        st.metric("Total activities detected", total)
        st.metric("Fall Detected count", counts.get("Fall Detected", 0))
        st.metric(
            "Normal activity count",
            sum(v for k, v in counts.items() if k != "Fall Detected"),
        )
        if confidences:
            st.metric("Avg. prediction confidence", f"{np.mean(confidences)*100:.1f}%")
 
        if counts:
            dist_df = pd.DataFrame(
                {"activity": list(counts.keys()), "count": list(counts.values())}
            ).set_index("activity")
            st.bar_chart(dist_df)
 
with tab_performance:
    st.subheader("Trained Model — Evaluation Evidence")
    st.caption(
        "Generated by 4_evaluate_model.py from the held-out test split. "
        "Copy these files into the model/ folder to display them here."
    )
 
    hist_path = os.path.join(MODEL_DIR, "training_history.png")
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix_nn.png")
    report_path = os.path.join(MODEL_DIR, "nn_classification_report.txt")
 
    c1, c2 = st.columns(2)
    if os.path.exists(hist_path):
        c1.image(hist_path, caption="Training accuracy & loss curves")
    else:
        c1.info("training_history.png not found in model/ — run 3_train_model.py.")
 
    if os.path.exists(cm_path):
        c2.image(cm_path, caption="Confusion matrix (test set)")
    else:
        c2.info("confusion_matrix_nn.png not found in model/ — run 4_evaluate_model.py.")
 
    if os.path.exists(report_path):
        with open(report_path) as f:
            st.text(f.read())
    else:
        st.info("nn_classification_report.txt not found in model/.")
 
