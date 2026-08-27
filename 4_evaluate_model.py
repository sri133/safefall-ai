import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)

from utils import FEATURE_COLUMNS


def main(model_dir):
    nn_model = tf.keras.models.load_model(os.path.join(model_dir, "model_nn.keras"))
    rf_model = joblib.load(os.path.join(model_dir, "model_rf.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    df_test = pd.read_csv(os.path.join(model_dir, "test_split.csv"))

    X_test = df_test[FEATURE_COLUMNS].values.astype(np.float32)
    y_test = le.transform(df_test["label"].values)
    X_test_s = scaler.transform(X_test)

    
    y_prob = nn_model.predict(X_test_s, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Deep Learning Model (Dense NN) — Test Set ===")
    print(f"Accuracy: {acc:.4f}\n")
    report = classification_report(y_test, y_pred, target_names=le.classes_, digits=3)
    print(report)

    with open(os.path.join(model_dir, "nn_classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Fall vs Not-Fall (Dense NN)")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix_nn.png"), dpi=150)
    print(f"Saved confusion_matrix_nn.png")

    
    y_pred_rf = rf_model.predict(X_test_s)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"\n=== Random Forest Baseline — Test Set ===")
    print(f"Accuracy: {acc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_, digits=3))

    print("\n=== Notes on deployment challenges (discuss these in your report) ===")
    print("- Lighting variation across scenes (Coffee_room vs Home vs Office)")
    print("- Camera angle differences affect landmark visibility/confidence")
    print("- Occlusion by furniture can hide hip/knee/ankle joints")
    print("- Fast falls sampled too coarsely (low sample_every) can be missed")
    print("- Similar postures (crouching vs sitting) can be confused by the")
    print("  rule-based sub-classifier used for Walking/Sitting/Standing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()
    main(args.model_dir)
