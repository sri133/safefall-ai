"""
1_extract_frames_and_labels.py

Step 4/5 groundwork: turn the raw Le2i / IMVIA fall dataset into labelled
image frames.

EXPECTED DATASET LAYOUT (this is how the Kaggle "falldataset-imvia" set is
organised — adjust SCENE_FOLDERS below if yours differs slightly):

    dataset_root/
        Coffee_room_01/
            Videos/            *.avi
            Annotation_files/  *.txt   (same stem name as the video)
        Coffee_room_02/...
        Home_01/...
        Home_02/...
        Lecture_room/...
        Office/...

ANNOTATION FORMAT (standard Le2i format):
    line 1: frame number where the fall STARTS  (0 if no fall in this video)
    line 2: frame number where the fall ENDS    (0 if no fall in this video)
    remaining lines: per-frame bounding box info (not used here)

We sample every Nth frame from each video, and label each sampled frame:
    - "fall"      if its frame index falls within [start, end]
    - "not_fall"  otherwise

Run in Colab:
    !python 1_extract_frames_and_labels.py \
        --dataset_root /content/falldataset-imvia \
        --out_dir /content/frames \
        --sample_every 5
"""

import argparse
import csv
import os
from pathlib import Path

import cv2

SCENE_FOLDERS = [
    "Coffee_room_01", "Coffee_room_02",
    "Home_01", "Home_02",
    "Lecture_room", "Office",
]


def read_annotation(ann_path):
    """Return (start_frame, end_frame) as ints. (0, 0) means no fall."""
    with open(ann_path, "r", errors="ignore") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() != ""]
    try:
        start = int(float(lines[0]))
        end = int(float(lines[1]))
    except (IndexError, ValueError):
        start, end = 0, 0
    return start, end


def find_pairs(dataset_root):
    """Yield (video_path, annotation_path, scene_name) for every video found."""
    dataset_root = Path(dataset_root)
    scene_dirs = [d for d in dataset_root.iterdir() if d.is_dir()] \
        if not any((dataset_root / s).exists() for s in SCENE_FOLDERS) \
        else [dataset_root / s for s in SCENE_FOLDERS if (dataset_root / s).exists()]

    for scene_dir in scene_dirs:
        video_dir = scene_dir / "Videos"
        ann_dir = scene_dir / "Annotation_files"
        if not video_dir.exists():
            # some Kaggle mirrors flatten the structure - search recursively
            video_dir = scene_dir
        if not ann_dir.exists():
            ann_dir = scene_dir

        for vid_path in video_dir.rglob("*.avi"):
            candidate = ann_dir / (vid_path.stem + ".txt")
            if not candidate.exists():
                # try a recursive search as a fallback
                matches = list(ann_dir.rglob(vid_path.stem + ".txt"))
                candidate = matches[0] if matches else None
            yield vid_path, candidate, scene_dir.name


def extract_frames(dataset_root, out_dir, sample_every=5, img_size=224):
    out_dir = Path(out_dir)
    (out_dir / "fall").mkdir(parents=True, exist_ok=True)
    (out_dir / "not_fall").mkdir(parents=True, exist_ok=True)

    rows = []
    pairs = list(find_pairs(dataset_root))
    print(f"Found {len(pairs)} video files.")

    for vid_path, ann_path, scene in pairs:
        if ann_path is None:
            print(f"  [skip] no annotation found for {vid_path.name}")
            continue
        start, end = read_annotation(ann_path)

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"  [skip] could not open {vid_path.name}")
            continue

        frame_idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_every != 0:
                continue

            is_fall = start > 0 and start <= frame_idx <= end
            label = "fall" if is_fall else "not_fall"

            frame_resized = cv2.resize(frame, (img_size, img_size))
            # blur / corruption check: skip near-black or near-white frames
            mean_val = frame_resized.mean()
            if mean_val < 5 or mean_val > 250:
                continue

            fname = f"{scene}_{vid_path.stem}_f{frame_idx:05d}.jpg"
            out_path = out_dir / label / fname
            cv2.imwrite(str(out_path), frame_resized)
            rows.append({
                "frame_path": str(out_path),
                "label": label,
                "scene": scene,
                "video": vid_path.stem,
                "frame_idx": frame_idx,
            })
            saved += 1

        cap.release()
        print(f"  {vid_path.name}: saved {saved} frames (fall window {start}-{end})")

    labels_csv = out_dir / "labels.csv"
    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "label", "scene", "video", "frame_idx"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} frames saved. Labels written to {labels_csv}")
    n_fall = sum(1 for r in rows if r["label"] == "fall")
    print(f"  fall: {n_fall}   not_fall: {len(rows) - n_fall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    extract_frames(args.dataset_root, args.out_dir, args.sample_every, args.img_size)
