import argparse

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

from utils import (
    get_pose_landmarks, landmarks_to_dict, extract_feature_vector,
    FEATURE_COLUMNS,
)

mp_pose = mp.solutions.pose


def main(labels_csv, out_csv, min_detection_confidence=0.5):
    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} labelled frames.")

    rows = []
    dropped = 0

    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=min_detection_confidence,
    ) as pose_estimator:
        for _, r in tqdm(df.iterrows(), total=len(df)):
            image = cv2.imread(r["frame_path"])
            if image is None:
                dropped += 1
                continue

            results = get_pose_landmarks(image, pose_estimator)
            if results is None:
                dropped += 1
                continue

            lm_dict = landmarks_to_dict(results)
            feats = extract_feature_vector(lm_dict)
            if feats is None:
                dropped += 1
                continue

            row = dict(zip(FEATURE_COLUMNS, feats))
            row["label"] = r["label"]
            row["scene"] = r["scene"]
            row["video"] = r["video"]
            row["frame_idx"] = r["frame_idx"]
            row["frame_path"] = r["frame_path"]
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out_df)} rows with pose landmarks to {out_csv}")
    print(f"Dropped {dropped} frames (no person / low confidence detection).")
    print(out_df["label"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    args = parser.parse_args()
    main(args.labels_csv, args.out_csv, args.min_detection_confidence)
