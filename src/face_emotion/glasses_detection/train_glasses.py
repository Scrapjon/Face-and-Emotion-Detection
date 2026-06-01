"""
train_glasses.py
----------------
Trains a compact CNN for binary glasses detection.
Sourcing from open GitHub repos - no manual dataset download required.
Run build_dataset.py first to generate the dataset.

For better accuracy, supplement or replace data/glasses_dataset/ with MeGlass:
    https://drive.google.com/file/d/1V0c8p6MOlSFY5R-Hu9LxYZYLXd8B8j9q
    (47,917 real labelled images, 14,832 glasses / 33,085 no-glasses)

Run from the project root:
    python src/face_emotion/glasses_detection/train_glasses.py
"""

import os
import numpy as np
import keras
from keras import layers, models, callbacks
from pathlib import Path

# ---- config ----
DATASET_DIR  = Path("data/glasses_dataset")
MODEL_PATH   = Path("src/models/glasses_model.h5")
IMG_SIZE     = (64, 64)    # small enough to train fast, big enough to see glasses shapes
BATCH_SIZE   = 32
EPOCHS       = 25
LR           = 1e-3
VAL_SPLIT    = 0.20
SEED         = 42

os.makedirs(MODEL_PATH.parent, exist_ok=True)


def load_datasets():
    common = dict(
        directory        = DATASET_DIR,
        labels           = "inferred",
        label_mode       = "binary",
        image_size       = IMG_SIZE,
        batch_size       = BATCH_SIZE,
        seed             = SEED,
        validation_split = VAL_SPLIT,
    )
    train_ds    = keras.utils.image_dataset_from_directory(subset="training",   **common)
    val_ds      = keras.utils.image_dataset_from_directory(subset="validation", **common)
    class_names = train_ds.class_names

    # alphabetical: glasses < no_glasses -> glasses=0, no_glasses=1
    glasses_idx = class_names.index("glasses") if "glasses" in class_names else 0
    print(f"Classes: {class_names}  (glasses_index={glasses_idx})")
    print(f"sigmoid < 0.5 -> class {glasses_idx} -> glasses detected")

    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomBrightness(0.12),
    ], name="aug")

    import tensorflow as tf

    def prep_train(img, lbl):
        img = aug(img, training=True)
        img = tf.cast(img, tf.float32) / 255.0
        return img, lbl

    def prep_val(img, lbl):
        img = tf.cast(img, tf.float32) / 255.0
        return img, lbl

    train_ds = train_ds.map(prep_train).prefetch(4)
    val_ds   = val_ds.map(prep_val).prefetch(4)

    return train_ds, val_ds, glasses_idx


def build_model():
    """Compact CNN for glasses detection - trains fast even on CPU."""
    inp = layers.Input((64, 64, 3))
    x   = layers.Conv2D(32,  3, padding="same", activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D()(x)          # 32x32

    x   = layers.Conv2D(64,  3, padding="same", activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D()(x)          # 16x16

    x   = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D()(x)          # 8x8

    x   = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalAveragePooling2D()(x)

    x   = layers.Dropout(0.40)(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.30)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inp, out, name="glasses_cnn")


def main():
    print("=" * 55)
    print("  Glasses Detection - Training")
    print("=" * 55)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"\nDataset not found at {DATASET_DIR}\n"
            "Run build_dataset.py first, or place a glasses/ and no_glasses/ folder there."
        )

    train_ds, val_ds, glasses_idx = load_datasets()
    model = build_model()
    print(f"Parameters: {model.count_params():,}")

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
        callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=0
        ),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    h = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=1
    )

    model.save(str(MODEL_PATH))
    best_acc = max(h.history["val_accuracy"])
    best_auc = max(h.history["val_auc"])
    print(f"\nBest val accuracy : {best_acc:.4f}")
    print(f"Best val AUC      : {best_auc:.4f}")
    print(f"Model saved       -> {MODEL_PATH}")
    print(f"      dataset with MeGlass: https://tinyurl.com/meglass-dataset")


if __name__ == "__main__":
    main()
