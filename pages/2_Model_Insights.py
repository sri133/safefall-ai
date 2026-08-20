"""
pages/2_Model_Insights.py
 
FA-2, Step 6 extra evaluation evidence.
 
Self-contained: uses the REAL precision/recall/F1/accuracy numbers from
4_evaluate_model.py's actual test-set run (hardcoded here), so this page
has no dependency on any file existing in the repo.
 
Three deliberately different chart types, each picked to fit what it's
showing:
  - Radar/spider chart  -> Precision/Recall/F1 per class (3 axes = a
                            natural triangle shape, easy to compare the
                            two classes' "footprint")
  - Line / slope chart  -> NN vs RF across three metrics, showing the
                            trade-off as two diverging lines
  - Funnel / chevron     -> key headline metrics ranked top to bottom,
                            KPI-dashboard style
"""
 
import plotly.graph_objects as go
import streamlit as st
 
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_CYAN = "#7df9ff"
ACCENT_RED = "#ff3860"
ACCENT_GREEN = "#00e676"
ACCENT_AMBER = "#ffb703"
ACCENT_PURPLE = "#a855f7"
 
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
 
# ============================================================
# CHART 1: Radar / spider (triangle) -- Precision/Recall/F1 per class
# ============================================================
st.header("Precision · Recall · F1 — Class Footprint")
st.caption("Each class's shape across the three metrics -- a balanced model would look like an even triangle; a lopsided shape reveals a trade-off, exactly what we see for the fall class.")
 
categories = ["Precision", "Recall", "F1-score"]
fall_vals = [nn_metrics["precision"]["fall"], nn_metrics["recall"]["fall"], nn_metrics["f1"]["fall"]]
notfall_vals = [nn_metrics["precision"]["not_fall"], nn_metrics["recall"]["not_fall"], nn_metrics["f1"]["not_fall"]]
 
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=fall_vals + [fall_vals[0]], theta=categories + [categories[0]],
    fill="toself", name="Fall class", line=dict(color=ACCENT_RED, width=2),
    fillcolor="rgba(255,56,96,0.25)",
))
fig.add_trace(go.Scatterpolar(
    r=notfall_vals + [notfall_vals[0]], theta=categories + [categories[0]],
    fill="toself", name="Not-fall class", line=dict(color=ACCENT_GREEN, width=2),
    fillcolor="rgba(0,230,118,0.20)",
))
fig.update_layout(
    template=PLOTLY_TEMPLATE, height=440,
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    margin=dict(l=40, r=40, t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)
 
# ============================================================
# CHART 2: Line / slope chart -- NN vs RF trade-off
# ============================================================
st.header("Dense NN vs Random Forest — The Trade-off")
st.caption(
    "Random Forest wins on raw accuracy, but its fall-recall line drops "
    "sharply -- meaning it misses far more real falls. This is why the "
    "deployed app uses the Dense NN despite its lower accuracy."
)
 
metric_names = ["Accuracy", "Fall Recall", "Fall Precision"]
nn_line = [nn_accuracy, nn_metrics["recall"]["fall"], nn_metrics["precision"]["fall"]]
rf_line = [rf_accuracy, rf_metrics["recall"]["fall"], rf_metrics["precision"]["fall"]]
 
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=metric_names, y=nn_line, mode="lines+markers+text", name="Dense NN",
    line=dict(color=ACCENT_CYAN, width=4), marker=dict(size=12),
    text=[f"{v*100:.1f}%" for v in nn_line], textposition="top center",
))
fig.add_trace(go.Scatter(
    x=metric_names, y=rf_line, mode="lines+markers+text", name="Random Forest",
    line=dict(color=ACCENT_AMBER, width=4, dash="dash"), marker=dict(size=12),
    text=[f"{v*100:.1f}%" for v in rf_line], textposition="bottom center",
))
fig.update_layout(
    template=PLOTLY_TEMPLATE, height=420, yaxis_range=[0, 1.15],
    yaxis_title="Score", margin=dict(l=40, r=20, t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)
 
c1, c2, c3 = st.columns(3)
c1.metric("Dense NN — Accuracy", f"{nn_accuracy*100:.1f}%")
c2.metric("Dense NN — Fall Recall", f"{nn_metrics['recall']['fall']*100:.1f}%",
          delta=f"+{(nn_metrics['recall']['fall']-rf_metrics['recall']['fall'])*100:.1f}% vs RF")
c3.metric("Random Forest — Accuracy", f"{rf_accuracy*100:.1f}%")
 
# ============================================================
# CHART 3: Funnel / chevron -- key metrics ranked
# ============================================================
st.header("Key Metrics — Ranked Overview")
st.caption("Headline numbers for the deployed Dense NN model, ranked highest to lowest.")
 
funnel_labels = ["Not-Fall Recall", "Overall Accuracy", "Not-Fall F1", "Fall Recall", "Fall Precision"]
funnel_values = [
    nn_metrics["recall"]["not_fall"] * 100,
    nn_accuracy * 100,
    nn_metrics["f1"]["not_fall"] * 100,
    nn_metrics["recall"]["fall"] * 100,
    nn_metrics["precision"]["fall"] * 100,
]
funnel_colors = [ACCENT_GREEN, ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_RED]
 
fig = go.Figure(go.Funnel(
    y=funnel_labels, x=funnel_values,
    textinfo="value+label", texttemplate="%{label}<br><b>%{value:.1f}%</b>",
    marker=dict(color=funnel_colors),
    connector=dict(line=dict(color=ACCENT_PURPLE, width=1)),
))
fig.update_layout(template=PLOTLY_TEMPLATE, height=440, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)
 
st.info(
    "**Why this matters:** Random Forest looks better on paper (95.1% "
    "accuracy vs 88.9%), but it only catches 17.3% of real falls, "
    "compared to 59.6% for the Dense NN. The dataset's heavy class "
    "imbalance (only ~5.7% of frames are falls) means a model can score "
    "high on accuracy while being nearly useless for the one thing this "
    "system needs to do. This is why the deployed app uses the Dense NN, "
    "and why recall — not accuracy — was the deciding metric."
)
 
