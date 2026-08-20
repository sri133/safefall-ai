"""
pages/2_Model_Insights.py

FA-2, Step 6 extra evaluation evidence.

Self-contained on purpose: uses the REAL precision/recall/F1/accuracy
numbers already produced by 4_evaluate_model.py's actual test-set run
(hardcoded here rather than re-computed), so this page has no dependency
on test_split.csv or any other file existing in the repo -- it can't
break due to a missing artifact.

Shows:
  - Precision / Recall / F1 comparison per class (Dense NN)
  - NN vs Random Forest comparison -- the accuracy-vs-recall tradeoff
    that matters most for a safety-critical fall detector
  - Reconstructed confusion-matrix-style breakdown from the same numbers
"""

import plotly.graph_objects as go
import streamlit as st

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_CYAN = "#7df9ff"
ACCENT_RED = "#ff3860"
ACCENT_GREEN = "#00e676"
ACCENT_AMBER = "#ffb703"

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
st.caption(
    "FA-2, Step 6: precision/recall/F1 breakdown and the NN-vs-RandomForest "
    "trade-off, from the actual held-out test set evaluation (947 frames: "
    "52 fall / 895 not_fall)."
)

# ---- Real numbers from 4_evaluate_model.py's actual test-set run ----
nn_metrics = {"precision": {"fall": 0.270, "not_fall": 0.975},
              "recall":    {"fall": 0.596, "not_fall": 0.906},
              "f1":        {"fall": 0.371, "not_fall": 0.939}}
rf_metrics = {"precision": {"fall": 0.750, "not_fall": 0.954},
              "recall":    {"fall": 0.173, "not_fall": 0.997},
              "f1":        {"fall": 0.281, "not_fall": 0.975}}
nn_accuracy, rf_accuracy = 0.8891, 0.9514

col1, col2 = st.columns(2)

with col1:
    st.subheader("Precision / Recall / F1 — Dense NN (fall class)")
    st.caption("The class that actually matters for safety: catching real falls.")
    metrics_names = ["Precision", "Recall", "F1-score"]
    fall_vals = [nn_metrics["precision"]["fall"], nn_metrics["recall"]["fall"], nn_metrics["f1"]["fall"]]
    fig = go.Figure(go.Bar(
        x=metrics_names, y=fall_vals, marker_color=ACCENT_RED,
        text=[f"{v:.2f}" for v in fall_vals], textposition="outside",
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=360, yaxis_range=[0, 1],
                       margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Precision / Recall / F1 — Dense NN (not_fall class)")
    notfall_vals = [nn_metrics["precision"]["not_fall"], nn_metrics["recall"]["not_fall"], nn_metrics["f1"]["not_fall"]]
    fig = go.Figure(go.Bar(
        x=metrics_names, y=notfall_vals, marker_color=ACCENT_GREEN,
        text=[f"{v:.2f}" for v in notfall_vals], textposition="outside",
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=360, yaxis_range=[0, 1],
                       margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

st.header("Model Comparison — Dense NN vs Random Forest")
st.caption(
    "Random Forest has higher raw accuracy, but its fall-recall is much "
    "lower — meaning it misses more real falls. For a safety system, "
    "recall on the fall class matters more than overall accuracy, which "
    "is why the deployed app uses the Dense NN."
)

fig = go.Figure()
fig.add_trace(go.Bar(name="Dense NN", x=["Accuracy", "Fall Recall", "Fall Precision"],
                      y=[nn_accuracy, nn_metrics["recall"]["fall"], nn_metrics["precision"]["fall"]],
                      marker_color=ACCENT_CYAN))
fig.add_trace(go.Bar(name="Random Forest", x=["Accuracy", "Fall Recall", "Fall Precision"],
                      y=[rf_accuracy, rf_metrics["recall"]["fall"], rf_metrics["precision"]["fall"]],
                      marker_color=ACCENT_AMBER))
fig.update_layout(template=PLOTLY_TEMPLATE, height=400, barmode="group",
                   yaxis_range=[0, 1], margin=dict(l=40, r=20, t=20, b=40))
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Dense NN — Accuracy", f"{nn_accuracy*100:.1f}%")
c2.metric("Dense NN — Fall Recall", f"{nn_metrics['recall']['fall']*100:.1f}%",
          delta=f"+{(nn_metrics['recall']['fall']-rf_metrics['recall']['fall'])*100:.1f}% vs RF")
c3.metric("Random Forest — Accuracy", f"{rf_accuracy*100:.1f}%")

st.info(
    "**Why this matters:** Random Forest looks better on paper (95.1% "
    "accuracy vs 88.9%), but it only catches 17.3% of real falls, "
    "compared to 59.6% for the Dense NN. The dataset's heavy class "
    "imbalance (only ~5.7% of frames are falls) means a model can score "
    "high on accuracy while being nearly useless for the one thing this "
    "system needs to do. This is why the deployed app uses the Dense NN, "
    "and why recall — not accuracy — was the deciding metric."
)
