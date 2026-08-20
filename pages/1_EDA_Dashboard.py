"""
pages/1_EDA_Dashboard.py

FA-1, Step 3 evidence, rendered live in the deployed app.

Reads the small summary CSVs produced once by 5_export_eda_summary.py
(committed to the repo under eda_data/) and renders:
  - Bar chart of frame counts per activity
  - Pie chart of class distribution
  - Activity distribution by scene
  - Summary table

This is a separate page automatically because it lives in pages/ --
Streamlit shows it in the sidebar navigation next to the main app.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

EDA_DIR = os.environ.get("EDA_DIR", "eda_data")

st.set_page_config(page_title="EDA Dashboard — SafeFall AI", layout="wide")

st.title("📊 Exploratory Data Analysis — Le2i Fall Dataset")
st.caption(
    "FA-1, Step 3: dataset composition and class balance for the elderly "
    "fall detection dataset used to train SafeFall AI."
)

overall_path = os.path.join(EDA_DIR, "eda_overall.csv")
scene_path = os.path.join(EDA_DIR, "eda_by_scene.csv")

if not os.path.exists(overall_path):
    st.error(
        f"Could not find '{overall_path}'. Run 5_export_eda_summary.py in "
        f"Colab and commit the resulting eda_data/ folder to your repo."
    )
    st.stop()

overall_df = pd.read_csv(overall_path)

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
    st.pyplot(fig)

with col2:
    st.subheader("Pie Chart — Class Distribution (%)")
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#e74c3c" if a == "fall" else "#2ecc71" for a in overall_df["activity"]]
    ax.pie(
        overall_df["frame_count"], labels=overall_df["activity"],
        autopct="%1.1f%%", colors=colors, startangle=90,
    )
    st.pyplot(fig)

st.header("Summary Table")
st.dataframe(overall_df, use_container_width=True)

total_frames = overall_df["frame_count"].sum()
imbalance_ratio = overall_df["frame_count"].max() / overall_df["frame_count"].min()
c1, c2, c3 = st.columns(3)
c1.metric("Total frames", int(total_frames))
c2.metric("Classes", len(overall_df))
c3.metric("Imbalance ratio", f"{imbalance_ratio:.1f} : 1")

if os.path.exists(scene_path):
    st.header("Activity Distribution by Scene")
    st.caption(
        "Breakdown across the Le2i scene folders (Coffee_room, Home, "
        "Lecture_room, Office) — useful for discussing camera angle and "
        "lighting variation across environments."
    )
    scene_df = pd.read_csv(scene_path)
    scene_df_indexed = scene_df.set_index(scene_df.columns[0])

    fig, ax = plt.subplots(figsize=(9, 5))
    scene_df_indexed.plot(kind="bar", stacked=True, ax=ax, color=["#e74c3c", "#2ecc71"])
    ax.set_xlabel("Scene")
    ax.set_ylabel("Frame Count")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

    st.dataframe(scene_df, use_container_width=True)
else:
    st.info(
        "Per-scene breakdown not found. Re-run 5_export_eda_summary.py on a "
        "labels.csv that includes a 'scene' column to enable this section."
    )
