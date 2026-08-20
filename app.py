"""
app.py
 
Step 7: Streamlit deployment -- SafeFall AI monitoring dashboard.
 
Dark "control room" theme with Plotly-based visuals:
  - Pose skeleton overlay + prediction on uploaded image/video
  - Live joint-angle radar chart for the current prediction
  - Fall-probability-over-time line chart (video mode)
  - Monitoring analytics: activity counts, distribution chart
  - Emergency alert banner with a pulse animation
"""
 
import os
import tempfile
from collections import Counter
 
import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
 
from utils import (
    get_pose_landmarks, landmarks_to_dict, extract_feature_vector,
    sub_classify_activity, draw_pose_and_label, ankle_midpoint,
    is_fall_alert, is_lying_down, compute_radar_features, FEATURE_COLUMNS,
)
import mediapipe as mp
 
MODEL_DIR = os.environ.get("MODEL_DIR", "model")
 
st.set_page_config(page_title="SafeFall AI — Elderly Monitoring", layout="wide")
 
# ---------------- Dark "control room" theme ----------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 20% 0%, #0f1729 0%, #05070d 60%);
    color: #e6f1ff;
}
h1, h2, h3 { color: #7df9ff !important; text-shadow: 0 0 12px rgba(125,249,255,0.35); }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #101a2e, #0a1120);
    border: 1px solid rgba(125,249,255,0.25);
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 0 18px rgba(125,249,255,0.08);
}
[data-testid="stMetricLabel"] { color: #8fa3c7 !important; }
[data-testid="stMetricValue"] { color: #7df9ff !important; }
.alert-banner {
    background: linear-gradient(90deg, #ff004080, #ff004033);
    border: 1px solid #ff3860;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 1.05rem;
    font-weight: 600;
    color: #ffe3ea;
    animation: pulse 1.4s infinite;
    box-shadow: 0 0 25px rgba(255,56,96,0.5);
}
.safe-banner {
    background: linear-gradient(90deg, #00c85333, #00c8531a);
    border: 1px solid #00e676;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 1.0rem;
    color: #d6ffe8;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 10px rgba(255,56,96,0.4); }
    50%  { box-shadow: 0 0 35px rgba(255,56,96,0.9); }
    100% { box-shadow: 0 0 10px rgba(255,56,96,0.4); }
}
</style>
""", unsafe_allow_html=True)
 
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_RED = "#ff3860"
ACCENT_GREEN = "#00e676"
ACCENT_CYAN = "#7df9ff"
 
 
@st.cache_resource
def load_artifacts():
    nn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_nn.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    return nn_model, scaler, le
 
 
def predict_frame(frame_bgr, pose_estimator, nn_model, scaler, le, prev_ankle=None):
    """Returns dict with label, confidence, annotated frame, fall probability,
    landmark dict (for radar chart), and new ankle position -- or None."""
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
    fall_idx = list(le.classes_).index("fall")
    fall_prob = float(probs[fall_idx])
 
    ankle_pos = ankle_midpoint(lm_dict)
    ankle_motion = None
    if prev_ankle is not None:
        ankle_motion = float(np.linalg.norm(ankle_pos - prev_ankle))
 
    if raw_label == "fall":
        final_label = "Fall Detected"
    elif is_lying_down(lm_dict):
        final_label = "Fall Detected"
        confidence = max(confidence, 0.75)
        fall_prob = max(fall_prob, 0.75)
    else:
        final_label = sub_classify_activity(lm_dict, ankle_motion=ankle_motion)
 
    alert = is_fall_alert(final_label, confidence)
    annotated = draw_pose_and_label(frame_bgr, results, final_label, confidence, alert)
 
    return {
        "label": final_label, "confidence": confidence, "annotated": annotated,
        "fall_prob": fall_prob, "lm_dict": lm_dict, "ankle_pos": ankle_pos,
    }
 
 
def radar_chart(radar_dict, label):
    categories = list(radar_dict.keys())
    values = list(radar_dict.values())
    is_fall = (label == "Fall Detected")
    line_color = ACCENT_RED if is_fall else ACCENT_CYAN
    fill_color = "rgba(255,56,96,0.25)" if is_fall else "rgba(125,249,255,0.20)"
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill="toself", line=dict(color=line_color, width=2),
        fillcolor=fill_color,
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, showlegend=False,
        polar=dict(radialaxis=dict(visible=True, range=[0, 180])),
        margin=dict(l=30, r=30, t=30, b=30), height=340,
        title="Live Joint-Angle Posture Radar",
    )
    return fig
 
 
def probability_timeline_chart(prob_history, alert_threshold=0.6):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=prob_history, mode="lines", name="Fall Probability",
        line=dict(color=ACCENT_RED, width=2),
        fill="tozeroy", fillcolor="rgba(255,56,96,0.15)",
    ))
    fig.add_hline(
        y=alert_threshold, line_dash="dash", line_color=ACCENT_CYAN,
        annotation_text="Alert Threshold", annotation_position="top left",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=320,
        title="Fall Probability Over Time",
        xaxis_title="Sampled Frame Index", yaxis_title="P(Fall)",
        yaxis_range=[0, 1], margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
 
 
def distribution_bar(counts):
    fig = go.Figure(go.Bar(
        x=list(counts.keys()), y=list(counts.values()),
        marker_color=[ACCENT_RED if k == "Fall Detected" else ACCENT_GREEN for k in counts.keys()],
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=300, title="Activity Distribution",
        margin=dict(l=30, r=20, t=40, b=30),
    )
    return fig
 
 
# ---------------- UI ----------------
st.title("🏥 SafeFall AI — Elderly Fall Detection & Monitoring")
st.caption(
    "CareVision HealthTech — Computer Vision based elderly activity monitoring. "
    "Pose estimation: MediaPipe Pose. Classifier: Dense Neural Network on pose landmarks."
)
 
try:
    nn_model, scaler, le = load_artifacts()
except Exception as e:
    st.error(
        f"Could not load model artifacts from '{MODEL_DIR}'. Make sure "
        f"model_nn.keras, scaler.joblib and label_encoder.joblib exist there.\n\nError: {e}"
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
                counts[out["label"]] += 1
                confidences.append(out["confidence"])
 
                with results_area:
                    c1, c2 = st.columns(2)
                    c1.image(
                        cv2.cvtColor(out["annotated"], cv2.COLOR_BGR2RGB),
                        caption="Pose visualization + prediction",
                    )
                    c1.metric("Predicted Activity", out["label"])
                    c1.metric("Confidence", f"{out['confidence']*100:.1f}%")
 
                    radar_dict = compute_radar_features(out["lm_dict"])
                    c2.plotly_chart(radar_chart(radar_dict, out["label"]), use_container_width=True)
 
                if is_fall_alert(out["label"], out["confidence"]):
                    alert_placeholder.markdown(
                        '<div class="alert-banner">🚨 EMERGENCY ALERT: Fall detected! '
                        'Notify caregiver immediately.</div>', unsafe_allow_html=True,
                    )
                else:
                    alert_placeholder.markdown(
                        '<div class="safe-banner">✅ No fall detected.</div>',
                        unsafe_allow_html=True,
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
                prob_history = []
                prev_ankle = None
                idx = 0
                fall_events = []
                last_radar = None
                last_label = None
 
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
 
                        prev_ankle = out["ankle_pos"]
                        counts[out["label"]] += 1
                        confidences.append(out["confidence"])
                        prob_history.append(out["fall_prob"])
                        last_radar = compute_radar_features(out["lm_dict"])
                        last_label = out["label"]
 
                        if is_fall_alert(out["label"], out["confidence"]):
                            fall_events.append(idx)
 
                        if len(frame_gallery) < 12:
                            frame_gallery.append((out["annotated"], out["label"], out["confidence"], idx))
 
                cap.release()
                progress.empty()
 
                if fall_events:
                    alert_placeholder.markdown(
                        f'<div class="alert-banner">🚨 EMERGENCY ALERT: Fall detected at frame(s) '
                        f'{fall_events[:5]}{"..." if len(fall_events) > 5 else ""}. '
                        f'Notify caregiver immediately.</div>', unsafe_allow_html=True,
                    )
                else:
                    alert_placeholder.markdown(
                        '<div class="safe-banner">✅ No falls detected in this video.</div>',
                        unsafe_allow_html=True,
                    )
 
                with results_area:
                    if prob_history:
                        st.plotly_chart(probability_timeline_chart(prob_history), use_container_width=True)
 
                    if last_radar:
                        st.plotly_chart(radar_chart(last_radar, last_label), use_container_width=True)
 
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
            st.plotly_chart(distribution_bar(counts), use_container_width=True)
 
with tab_performance:
    st.subheader("Trained Model — Evaluation Evidence")
    st.caption("Generated by 4_evaluate_model.py from the held-out test split.")
 
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
 
    st.info("For ROC curve, Precision-Recall curve, and per-scene accuracy, see the "
            "**Model Insights** page in the sidebar.")
 
