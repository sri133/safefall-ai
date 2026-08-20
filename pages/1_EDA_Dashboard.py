"""
pages/1_EDA_Dashboard.py
 
FA-1, Step 3 evidence, rendered live in the deployed Streamlit app.
 
Self-contained on purpose: the activity counts below are hardcoded from
the actual dataset run (1_extract_frames_and_labels.py output), so this
page needs NO external CSV file, NO Colab syncing step, and can't break
due to a missing data file. Just edit the numbers below if you re-run
extraction with different settings later.
 
This appears as a separate page automatically because it lives in
pages/ -- Streamlit shows it in the sidebar navigation next to the main
app, no extra setup required.
"""
 
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
 
st.set_page_config(page_title="EDA Dashboard — SafeFall AI", layout="wide")
 
st.title("📊 Exploratory Data Analysis — Le2i Fall Dataset")
st.caption(
    "FA-1, Step 3: dataset composition and class balance for the elderly "
    "fall detection dataset used to train SafeFall AI."
)
 
# ---- Real counts from the actual dataset run ----
# From 1_extract_frames_and_labels.py (raw extracted frames, before pose
# filtering): 8,366 total frames sampled from 190 videos across 6 scenes.
overall_data = {
    "activity": ["fall", "not_fall"],
    "frame_count": [456, 7910],
}
overall_df = pd.DataFrame(overall_data)
overall_df["percentage"] = (overall_df["frame_count"] / overall_df["frame_count"].sum() * 100).round(2)
 
st.header("Activity Class Distribution")
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("Bar Chart — Frame Count per Activity")
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#e74c3c" if a == "fall" else "#2ecc71" for a in overall_df["activity"]]
    ax.bar(overall_df["activity"], overall_df["frame_count"], color=colors)
    for i, v in enumerate(overall_df["frame_count"]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Frame Count")
    ax.set_title("Frames per Activity Class")
    st.pyplot(fig)
 
with col2:
    st.subheader("Pie Chart — Class Distribution (%)")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        overall_df["frame_count"], labels=overall_df["activity"],
        autopct="%1.1f%%", colors=colors, startangle=90,
    )
    ax.set_title("Class Distribution")
    st.pyplot(fig)
 
st.header("Summary Table")
st.dataframe(overall_df, use_container_width=True)
 
total_frames = overall_df["frame_count"].sum()
imbalance_ratio = overall_df["frame_count"].max() / overall_df["frame_count"].min()
c1, c2, c3 = st.columns(3)
c1.metric("Total frames extracted", int(total_frames))
c2.metric("Videos processed", 190)
c3.metric("Imbalance ratio", f"{imbalance_ratio:.1f} : 1")
 
st.header("Activity Distribution")
st.caption(
    "Frame-level activity distribution across the full extracted dataset. "
    "The dataset is heavily skewed toward not_fall frames, since a fall "
    "event only lasts a couple of seconds within each longer video clip — "
    "this imbalance was addressed during training using class weighting."
)
dist_chart_df = overall_df.set_index("activity")[["frame_count"]]
st.bar_chart(dist_chart_df)
 
st.info(
    "Note: annotations in the Le2i dataset mark only the fall event "
    "window itself (start/end frame), not per-frame labels for walking, "
    "sitting, or standing separately — so this EDA reflects a binary "
    "fall / not_fall breakdown, which is what the trained classifier "
    "also uses as ground truth (see FA-2 model training)."
)
 
