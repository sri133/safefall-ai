"""
pages/2_Model_Insights.py

FA-2, Step 6 extra evaluation evidence, rendered live from the actual
trained model and held-out test split (model/test_split.csv), computed
fresh on every page load -- not pre-baked images.

Adds three graphs beyond the confusion matrix already shown on the main
page:
  - ROC curve with AUC score
  - Precision-Recall curve (more informative than ROC given the class
    imbalance in this dataset)
  - Per-scene accuracy breakdown (Coffee_room / Home / Office / Lecture_room)
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score

from utils import FEATURE_COLUMNS

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_CYAN = "#7df9ff"
ACCENT_RED = "#ff3860"
ACCENT_GREEN = "#00e676"

st.set_page_config(page_title="Model Insights — SafeFall AI", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 20% 0%, #0f1729 0%, #05070d 60%); color: #e6f1ff; }
h1, h2, h3 { color: #7df9ff !important; text-shadow: 0 0 12px rgba(125,249,255,0.35); }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #101a2e, #0a1120);
    border: 1px solid rgba(125,249,255,0.25); border-radius: 12px; padding: 12px 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔬 Model Insights — Deep Evaluation")
st.caption("FA-2, Step 6: ROC curve, Precision-Recall curve, and per-scene accuracy, computed live from the held-out test set.")

try:
    nn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_nn.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    df_test = pd.read_csv(os.path.join(MODEL_DIR, "test_split.csv"))
except Exception as e:
    st.error(f"Could not load model/test artifacts from '{MODEL_DIR}'. Error: {e}")
    st.stop()

X_test = df_test[FEATURE_COLUMNS].values.astype(np.float32)
y_test_labels = df_test["label"].values
y_test = le.transform(y_test_labels)
X_test_s = scaler.transform(X_test)

y_prob = nn_model.predict(X_test_s, verbose=0)
fall_idx = list(le.classes_).index("fall")
y_prob_fall = y_prob[:, fall_idx]
y_true_fall = (y_test == fall_idx).astype(int)
y_pred = np.argmax(y_prob, axis=1)

col1, col2 = st.columns(2)

# ---- ROC curve ----
with col1:
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_true_fall, y_prob_fall)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.3f})",
                              line=dict(color=ACCENT_CYAN, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
                              line=dict(color="gray", dash="dash")))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                       xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                       margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.metric("AUC Score", f"{roc_auc:.3f}")

# ---- Precision-Recall curve ----
with col2:
    st.subheader("Precision-Recall Curve")
    st.caption("More informative than ROC here, given the heavy class imbalance (fall is the minority class).")
    precision, recall, _ = precision_recall_curve(y_true_fall, y_prob_fall)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Precision-Recall",
                              line=dict(color=ACCENT_RED, width=3), fill="tozeroy",
                              fillcolor="rgba(255,56,96,0.15)"))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                       xaxis_title="Recall", yaxis_title="Precision",
                       xaxis_range=[0, 1], yaxis_range=[0, 1.05],
                       margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

# ---- Per-scene accuracy ----
st.subheader("Per-Scene Accuracy")
st.caption("How well the model performs on each recording environment — relevant to the lighting/camera-angle deployment challenges discussed in the report.")

if "scene" in df_test.columns:
    df_test = df_test.copy()
    df_test["pred_idx"] = y_pred
    df_test["correct"] = (df_test["pred_idx"] == y_test)
    scene_acc = df_test.groupby("scene")["correct"].mean().sort_values(ascending=False)

    fig = go.Figure(go.Bar(
        x=scene_acc.index, y=scene_acc.values * 100,
        marker_color=ACCENT_GREEN,
        text=[f"{v*100:.1f}%" for v in scene_acc.values], textposition="outside",
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                       xaxis_title="Scene", yaxis_title="Accuracy (%)", yaxis_range=[0, 105],
                       margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Test split does not include a 'scene' column — re-run 3_train_model.py on a landmarks.csv built from labels.csv that includes scene info.")

overall_acc = accuracy_score(y_test, y_pred)
st.metric("Overall Test Accuracy", f"{overall_acc*100:.2f}%")
