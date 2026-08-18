"""
3_train_model.py

Step 5: Model Design and Training.

Trains:
  (A) A small deep learning classifier (Keras Dense network) on the pose
      landmark feature vectors -> Fall / Not-Fall.
  (B) A Random Forest baseline on the same features, for comparison
      (the rubric lists Random Forest as an acceptable model too -- having
      both strengthens your "Model Selection" writeup).

Split: 70% train / 15% val / 15% test, stratified by label.

Saves:
  model_nn.keras        - trained Keras model
  model_rf.joblib        - trained Random Forest
  scaler.joblib           - StandardScaler fit on train set
  training_history.png   - accuracy & loss curves
  test_split.csv          - held-out test rows (used by 4_evaluate_model.py)

Run in Colab:
    !python 3_train_model.py --landmarks_csv /content/landmarks.csv --out_dir /content/model
"""

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils import FEATURE_COLUMNS


def build_nn(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(landmarks_csv, out_dir, epochs=60, batch_size=32):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(landmarks_csv)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
    print(f"Loaded {len(df)} rows.")
    print(df["label"].value_counts())

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)  # fall / not_fall -> 0/1
    print("Classes:", list(le.classes_))

    # 70/15/15 stratified split
    X_train, X_temp, y_train, y_temp, df_train, df_temp = train_test_split(
        X, y, df, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test, df_val, df_test = train_test_split(
        X_temp, y_temp, df_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ---- (A) Deep learning model ----
    nn_model = build_nn(X_train_s.shape[1], num_classes=len(le.classes_))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    history = nn_model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    # Accuracy / loss curves (required evidence per rubric)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_history.png"), dpi=150)
    print(f"Saved training_history.png")

    nn_test_acc = nn_model.evaluate(X_test_s, y_test, verbose=0)[1]
    print(f"NN test accuracy: {nn_test_acc:.4f}")

    # ---- (B) Random Forest baseline ----
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    rf_model.fit(X_train_s, y_train)
    rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test_s))
    print(f"RF test accuracy: {rf_test_acc:.4f}")

    # ---- Save everything ----
    nn_model.save(os.path.join(out_dir, "model_nn.keras"))
    joblib.dump(rf_model, os.path.join(out_dir, "model_rf.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.joblib"))
    df_test.to_csv(os.path.join(out_dir, "test_split.csv"), index=False)

    print(f"\nAll artifacts saved to {out_dir}")
    print(f"  model_nn.keras, model_rf.joblib, scaler.joblib, label_encoder.joblib")
    print(f"  test_split.csv  (used by 4_evaluate_model.py)")
    print(f"\nSummary: NN test acc = {nn_test_acc:.4f} | RF test acc = {rf_test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args.landmarks_csv, args.out_dir, args.epochs, args.batch_size)
