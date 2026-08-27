import base64
import io
import os
import tempfile
import time
from collections import Counter
 
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
 
from utils import (
    get_pose_landmarks, landmarks_to_dict, extract_feature_vector,
    sub_classify_activity, draw_pose_and_label, ankle_midpoint,
    is_fall_alert, is_lying_down, compute_radar_features,
    generate_beep_base64, FEATURE_COLUMNS,
)
import mediapipe as mp
 
MODEL_DIR = os.environ.get("MODEL_DIR", "model")
 
st.set_page_config(page_title="SafeFall AI — Elderly Monitoring", layout="wide")
 

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
    border: 1px solid #ff3860; border-radius: 10px; padding: 14px 18px;
    font-size: 1.05rem; font-weight: 600; color: #ffe3ea;
    animation: pulse 1.4s infinite; box-shadow: 0 0 25px rgba(255,56,96,0.5);
}
.safe-banner {
    background: linear-gradient(90deg, #00c85333, #00c8531a);
    border: 1px solid #00e676; border-radius: 10px; padding: 14px 18px;
    font-size: 1.0rem; color: #d6ffe8;
}
.shift-card {
    background: linear-gradient(145deg, #131f38, #0a1120);
    border: 1px solid rgba(125,249,255,0.3); border-radius: 14px;
    padding: 20px 24px; box-shadow: 0 0 25px rgba(125,249,255,0.12);
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
ACCENT_AMBER = "#ffb703"
 
 
@st.cache_resource
def load_artifacts():
    nn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_nn.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    rf_model = None
    rf_path = os.path.join(MODEL_DIR, "model_rf.joblib")
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
    return nn_model, scaler, le, rf_model
 
 
def predict_frame(frame_bgr, pose_estimator, nn_model, scaler, le, prev_ankle=None, rf_model=None):
    """Returns a dict with NN prediction (+ optional RF comparison), the
    annotated frame, fall probability, landmarks, timing, and ankle pos."""
    t0 = time.perf_counter()
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
 
    result = {
        "label": final_label, "confidence": confidence, "annotated": annotated,
        "fall_prob": fall_prob, "lm_dict": lm_dict, "ankle_pos": ankle_pos,
    }
 
    
    if rf_model is not None:
        rf_probs = rf_model.predict_proba(feats_scaled)[0]
        rf_pred_idx = int(np.argmax(rf_probs))
        rf_raw_label = le.classes_[rf_pred_idx]
        rf_confidence = float(rf_probs[rf_pred_idx])
        rf_fall_prob = float(rf_probs[fall_idx])
        result["rf_label"] = "Fall Detected" if rf_raw_label == "fall" else "Not Fall"
        result["rf_confidence"] = rf_confidence
        result["rf_fall_prob"] = rf_fall_prob
 
    
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return result
 
 
def play_beep():
    """Feature 2: audible alert, generated in-memory, embedded as autoplay HTML5 audio."""
    b64 = generate_beep_base64()
    st.markdown(
        f'<audio autoplay="true"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )
 
 
def radar_chart(radar_dict, label):
    categories = list(radar_dict.keys())
    values = list(radar_dict.values())
    is_fall = (label == "Fall Detected")
    line_color = ACCENT_RED if is_fall else ACCENT_CYAN
    fill_color = "rgba(255,56,96,0.25)" if is_fall else "rgba(125,249,255,0.20)"
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill="toself", line=dict(color=line_color, width=2), fillcolor=fill_color,
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, showlegend=False,
        polar=dict(radialaxis=dict(visible=True, range=[0, 180])),
        margin=dict(l=30, r=30, t=30, b=30), height=340,
        title="Live Joint-Angle Posture Radar",
    )
    return fig
 
 
def probability_timeline_chart(nn_history, rf_history=None, alert_threshold=0.6):
    """Feature 5 extended into the timeline: NN vs RF fall-probability over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=nn_history, mode="lines", name="Dense NN",
        line=dict(color=ACCENT_RED, width=2), fill="tozeroy", fillcolor="rgba(255,56,96,0.12)",
    ))
    if rf_history:
        fig.add_trace(go.Scatter(
            y=rf_history, mode="lines", name="Random Forest",
            line=dict(color=ACCENT_AMBER, width=2, dash="dash"),
        ))
    fig.add_hline(y=alert_threshold, line_dash="dash", line_color=ACCENT_CYAN,
                  annotation_text="Alert Threshold", annotation_position="top left")
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=340, title="Fall Probability Over Time — NN vs RF",
        xaxis_title="Sampled Frame Index", yaxis_title="P(Fall)",
        yaxis_range=[0, 1], margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
 
 
def distribution_bar(counts):
    fig = go.Figure(go.Bar(
        x=list(counts.keys()), y=list(counts.values()),
        marker_color=[ACCENT_RED if k == "Fall Detected" else ACCENT_GREEN for k in counts.keys()],
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=300, title="Activity Distribution",
                       margin=dict(l=30, r=20, t=40, b=30))
    return fig
 
 
def build_pdf_report(counts, confidences, fall_events, avg_speed_ms, monitored_seconds, alert_log):
    """Feature 4: downloadable incident report (matplotlib -> PDF bytes)."""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.suptitle("SafeFall AI — Incident Report", fontsize=18, fontweight="bold", y=0.97)
 
    total = sum(counts.values())
    fall_count = counts.get("Fall Detected", 0)
    avg_conf = np.mean(confidences) * 100 if confidences else 0.0
 
    summary_lines = [
        f"Total frames analyzed: {total}",
        f"Monitored duration (approx): {monitored_seconds:.1f} seconds",
        f"Falls detected: {fall_count}",
        f"Fall alert frame indices: {fall_events[:10]}{'...' if len(fall_events) > 10 else ''}" if fall_events else "Fall alert frame indices: none",
        f"Average prediction confidence: {avg_conf:.1f}%",
        f"Average processing speed: {avg_speed_ms:.1f} ms/frame ({1000/avg_speed_ms:.1f} FPS equivalent)" if avg_speed_ms else "Average processing speed: n/a",
    ]
    fig.text(0.08, 0.88, "\n".join(summary_lines), fontsize=11, va="top", family="monospace")
 
    ax = fig.add_axes([0.12, 0.45, 0.76, 0.3])
    if counts:
        colors = ["#ff3860" if k == "Fall Detected" else "#2ecc71" for k in counts.keys()]
        ax.bar(list(counts.keys()), list(counts.values()), color=colors)
        ax.set_title("Activity Distribution")
        ax.set_ylabel("Frame Count")
 
    if alert_log:
        fig.text(0.08, 0.40, "Alert Log (frame index — confidence):", fontsize=11, fontweight="bold")
        log_text = "\n".join([f"  Frame {e['frame_idx']}  —  {e['confidence']*100:.1f}% confidence" for e in alert_log[:15]])
        fig.text(0.08, 0.36, log_text, fontsize=9, va="top", family="monospace")
 
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    buf.seek(0)
    return buf
 
 

st.title("🏥 SafeFall AI — Elderly Fall Detection & Monitoring")
st.caption(
    "CareVision HealthTech — Computer Vision based elderly activity monitoring. "
    "Pose estimation: MediaPipe Pose. Classifier: Dense Neural Network on pose landmarks."
)
 
try:
    nn_model, scaler, le, rf_model = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts from '{MODEL_DIR}'.\n\nError: {e}")
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
                out = predict_frame(frame, pose_estimator, nn_model, scaler, le, rf_model=rf_model)
 
            if out is None:
                st.warning("No person detected in this image.")
            else:
                counts[out["label"]] += 1
                confidences.append(out["confidence"])
 
                with results_area:
                    c1, c2 = st.columns(2)
                    c1.image(cv2.cvtColor(out["annotated"], cv2.COLOR_BGR2RGB), caption="Pose visualization + prediction")
                    c1.metric("Predicted Activity (NN)", out["label"])
                    c1.metric("Confidence", f"{out['confidence']*100:.1f}%")
                    c1.metric("Processing Speed", f"{out['elapsed_ms']:.1f} ms")
 
                    c2.plotly_chart(radar_chart(compute_radar_features(out["lm_dict"]), out["label"]), use_container_width=True)
 
                    if "rf_label" in out:
                        st.subheader("🥊 Model Comparison")
                        m1, m2 = st.columns(2)
                        m1.metric("Dense NN says", out["label"], f"{out['confidence']*100:.1f}% confident")
                        m2.metric("Random Forest says", out["rf_label"], f"{out['rf_confidence']*100:.1f}% confident")
 
                if is_fall_alert(out["label"], out["confidence"]):
                    alert_placeholder.markdown(
                        '<div class="alert-banner">🚨 EMERGENCY ALERT: Fall detected! Notify caregiver immediately.</div>',
                        unsafe_allow_html=True,
                    )
                    play_beep()
                else:
                    alert_placeholder.markdown('<div class="safe-banner">✅ No fall detected.</div>', unsafe_allow_html=True)
 
        else:  
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
 
            sample_every = st.slider("Analyse every Nth frame", 1, 15, 5)
            run_btn = st.button("Run analysis")
 
            if run_btn:
                progress = st.progress(0, text="Analysing video...")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
 
                frame_gallery = []
                alert_log = []          
                nn_prob_history = []
                rf_prob_history = []
                speed_samples = []      
                prev_ankle = None
                idx = 0
                fall_events = []
                last_radar, last_label = None, None
 
                with mp.solutions.pose.Pose(static_image_mode=True) as pose_estimator:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        idx += 1
                        if idx % sample_every != 0:
                            continue
 
                        out = predict_frame(frame, pose_estimator, nn_model, scaler, le, prev_ankle, rf_model=rf_model)
                        progress.progress(min(idx / total_frames, 1.0))
                        if out is None:
                            continue
 
                        prev_ankle = out["ankle_pos"]
                        counts[out["label"]] += 1
                        confidences.append(out["confidence"])
                        nn_prob_history.append(out["fall_prob"])
                        if "rf_fall_prob" in out:
                            rf_prob_history.append(out["rf_fall_prob"])
                        speed_samples.append(out["elapsed_ms"])
                        last_radar = compute_radar_features(out["lm_dict"])
                        last_label = out["label"]
 
                        if is_fall_alert(out["label"], out["confidence"]):
                            fall_events.append(idx)
                            alert_log.append({
                                "frame_idx": idx, "confidence": out["confidence"],
                                "thumbnail": out["annotated"],
                            })
 
                        if len(frame_gallery) < 12:
                            frame_gallery.append((out["annotated"], out["label"], out["confidence"], idx))
 
                cap.release()
                progress.empty()
 
                avg_speed_ms = float(np.mean(speed_samples)) if speed_samples else 0.0
                monitored_seconds = idx / source_fps if source_fps else 0.0
 
                if fall_events:
                    alert_placeholder.markdown(
                        f'<div class="alert-banner">🚨 EMERGENCY ALERT: Fall detected at frame(s) '
                        f'{fall_events[:5]}{"..." if len(fall_events) > 5 else ""}. '
                        f'Notify caregiver immediately.</div>', unsafe_allow_html=True,
                    )
                    play_beep()
                else:
                    alert_placeholder.markdown('<div class="safe-banner">✅ No falls detected in this video.</div>', unsafe_allow_html=True)
 
                with results_area:
                    
                    st.markdown('<div class="shift-card">', unsafe_allow_html=True)
                    st.subheader("📋 Shift Report")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Monitored duration", f"{monitored_seconds:.1f}s")
                    r2.metric("Falls detected", len(fall_events))
                    r3.metric("Avg. confidence", f"{np.mean(confidences)*100:.1f}%" if confidences else "n/a")
                    r4.metric("Avg. speed", f"{avg_speed_ms:.1f} ms/frame")
                    st.markdown('</div>', unsafe_allow_html=True)
 
                    
                    pdf_buf = build_pdf_report(counts, confidences, fall_events, avg_speed_ms, monitored_seconds, alert_log)
                    st.download_button(
                        "📄 Download Incident Report (PDF)", data=pdf_buf,
                        file_name="safefall_incident_report.pdf", mime="application/pdf",
                    )
 
                    
                    if nn_prob_history:
                        st.plotly_chart(probability_timeline_chart(nn_prob_history, rf_prob_history), use_container_width=True)
 
                    if last_radar:
                        st.plotly_chart(radar_chart(last_radar, last_label), use_container_width=True)
 
                    
                    if alert_log:
                        st.subheader("🚨 Alert History Log")
                        log_cols = st.columns(4)
                        for i, entry in enumerate(alert_log):
                            with log_cols[i % 4]:
                                st.image(cv2.cvtColor(entry["thumbnail"], cv2.COLOR_BGR2RGB),
                                         caption=f"Frame {entry['frame_idx']} — {entry['confidence']*100:.0f}%")
                    else:
                        st.info("No alerts logged during this session.")
 
                    st.subheader("Sampled frame predictions")
                    grid_cols = st.columns(4)
                    for i, (img, label, conf, frame_idx) in enumerate(frame_gallery):
                        with grid_cols[i % 4]:
                            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"frame {frame_idx}: {label} ({conf*100:.0f}%)")
 
    with col_stats:
        st.subheader("Monitoring Analytics")
        total = sum(counts.values())
        st.metric("Total activities detected", total)
        st.metric("Fall Detected count", counts.get("Fall Detected", 0))
        st.metric("Normal activity count", sum(v for k, v in counts.items() if k != "Fall Detected"))
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
 
    st.info("For ROC/PR-style breakdowns, see the **Model Insights** page in the sidebar.")
 
