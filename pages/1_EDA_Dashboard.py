import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_RED = "#ff3860"
ACCENT_GREEN = "#00e676"
ACCENT_CYAN = "#7df9ff"

st.set_page_config(page_title="EDA Dashboard — SafeFall AI", layout="wide")

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

st.title("📊 Exploratory Data Analysis — Le2i Fall Dataset")
st.caption("FA-1, Step 3: dataset composition and class balance for the elderly fall detection dataset used to train SafeFall AI.")

overall_df = pd.DataFrame({"activity": ["fall", "not_fall"], "frame_count": [456, 7910]})
overall_df["percentage"] = (overall_df["frame_count"] / overall_df["frame_count"].sum() * 100).round(2)
colors = [ACCENT_RED if a == "fall" else ACCENT_GREEN for a in overall_df["activity"]]

st.header("Activity Class Distribution")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Bar Chart — Frame Count per Activity")
    fig = go.Figure(go.Bar(
        x=overall_df["activity"], y=overall_df["frame_count"], marker_color=colors,
        text=overall_df["frame_count"], textposition="outside",
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                       xaxis_title="Activity", yaxis_title="Frame Count",
                       margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Pie Chart — Class Distribution (%)")
    fig = go.Figure(go.Pie(
        labels=overall_df["activity"], values=overall_df["frame_count"],
        marker=dict(colors=colors), hole=0.35,
        textinfo="label+percent",
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.header("Summary Table")
st.dataframe(overall_df, use_container_width=True)

total_frames = overall_df["frame_count"].sum()
imbalance_ratio = overall_df["frame_count"].max() / overall_df["frame_count"].min()
c1, c2, c3 = st.columns(3)
c1.metric("Total frames extracted", int(total_frames))
c2.metric("Videos processed", 190)
c3.metric("Imbalance ratio", f"{imbalance_ratio:.1f} : 1")

st.header("Activity Distribution")
st.caption("The dataset is heavily skewed toward not_fall frames, since a fall event only lasts a couple of seconds within each longer video clip — this imbalance was addressed during training using class weighting.")
fig = go.Figure(go.Bar(
    x=overall_df["activity"], y=overall_df["frame_count"], marker_color=colors, orientation="v",
))
fig.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(l=40, r=20, t=20, b=40))
st.plotly_chart(fig, use_container_width=True)

st.info(
    "Note: annotations in the Le2i dataset mark only the fall event window "
    "itself (start/end frame), not per-frame labels for walking, sitting, "
    "or standing separately — so this EDA reflects a binary fall / "
    "not_fall breakdown, which is what the trained classifier also uses "
    "as ground truth (see FA-2 model training)."
)
