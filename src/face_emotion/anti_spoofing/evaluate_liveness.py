from __future__ import annotations

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

MODEL_PATH = Path("models/liveness_model.keras")
VAL_DIR = Path("data/liveness_dataset/val")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )
    class_names = ds.class_names
    live_index = class_names.index("live")

    y_true = []
    y_prob = []

    for images, labels in ds:
        # Convert Keras alphabetical labels to live=1, spoof=0.
        labels = tf.cast(tf.equal(tf.cast(labels, tf.int32), live_index), tf.float32)
        probs = model.predict(images, verbose=0).reshape(-1)
        y_true.extend(labels.numpy().reshape(-1).tolist())
        y_prob.extend(probs.tolist())

    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.60).astype(int)

    print("Confusion matrix [[spoof_correct, spoof_missed], [live_blocked, live_correct]]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(classification_report(y_true, y_pred, target_names=["spoof", "live"]))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))


if __name__ == "__main__":
    main()
