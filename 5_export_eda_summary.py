"""
5_export_eda_summary.py

Run ONCE in Colab (wherever labels.csv already exists) to produce small
summary CSVs that are cheap enough to commit to GitHub and read directly
inside the Streamlit app -- no need to re-run any EDA/chart code in Colab.

Run in Colab:
    !python 5_export_eda_summary.py \
        --labels_csv /content/frames/labels.csv \
        --out_dir /content/eda_data
"""

import argparse
import os

import pandas as pd


def main(labels_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(labels_csv)

    # Overall activity counts
    overall = df["label"].value_counts().rename_axis("activity").reset_index(name="frame_count")
    overall["percentage"] = (overall["frame_count"] / len(df) * 100).round(2)
    overall_path = os.path.join(out_dir, "eda_overall.csv")
    overall.to_csv(overall_path, index=False)
    print(f"Saved {overall_path}")
    print(overall.to_string(index=False))

    # Per-scene breakdown (if the scene column exists)
    if "scene" in df.columns:
        scene = df.groupby(["scene", "label"]).size().unstack(fill_value=0).reset_index()
        scene_path = os.path.join(out_dir, "eda_by_scene.csv")
        scene.to_csv(scene_path, index=False)
        print(f"\nSaved {scene_path}")
        print(scene.to_string(index=False))

    print(f"\nTotal frames: {len(df)}")
    print(f"\nBoth files are tiny (a few KB) -- commit them to your GitHub repo")
    print(f"under an 'eda_data/' folder at the repo root.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.labels_csv, args.out_dir)
