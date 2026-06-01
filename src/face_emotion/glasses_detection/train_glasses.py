"""
train_glasses.py
----------------
Trains a glasses detector using MobileNetV2 transfer learning.
Two-phase training: frozen base (fast) -> fine-tune top layers (accurate).

Run build_dataset.py first, or use MeGlass for best results:
    https://drive.google.com/file/d/1V0c8p6MOlSFY5R-Hu9LxYZYLXd8B8j9q

Run from the project root:
    python src/face_emotion/glasses_detection/train_glasses.py
"""

import os
import numpy as np
import keras
from keras import layers, models, callbacks
from pathlib import Path

# ---- config ----
DATASET_DIR  = Path("src/data/glasses_dataset")
MODEL_PATH   = Path("src/models/glasses_model.h5")
IMG_SIZE     = (96, 96)    # MobileNetV2 minimum — must match glasses_model.py
BATCH_SIZE   = 32
SEED         = 42
VAL_SPLIT    = 0.20

# Phase 1: head-only training (base fully frozen)
PHASE1_EPOCHS = 15
PHASE1_LR     = 1e-3

# Phase 2: fine-tune top layers of MobileNetV2
PHASE2_EPOCHS  = 20
PHASE2_LR      = 1e-4   # 10x lower — avoid destroying ImageNet weights
FINE_TUNE_FROM = 100    # unfreeze layers from this index onward (~155 total in MobileNetV2)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

os.makedirs(MODEL_PATH.parent, exist_ok=True)


# ---- dataset helpers ----

def compute_class_weights(dataset_dir: Path, class_names: list) -> dict:
    counts = []
    for cls in class_names:
        cls_dir = dataset_dir / cls
        n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
        counts.append(n)
        print(f"  class '{cls}': {n} images")

    total     = sum(counts)
    n_classes = len(counts)
    weights   = {i: total / (n_classes * c) for i, c in enumerate(counts)}
    print(f"  -> class_weights: { {k: f'{v:.3f}' for k, v in weights.items()} }\n")
    return weights


def load_datasets():
    import tensorflow as tf

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

    # alphabetical: glasses=0, no_glasses=1
    glasses_idx = class_names.index("glasses") if "glasses" in class_names else 0
    print(f"Classes: {class_names}  (glasses_index={glasses_idx})")
    print(f"sigmoid < 0.5 -> class {glasses_idx} -> glasses detected\n")

    print("Computing class weights:")
    class_weights = compute_class_weights(DATASET_DIR, class_names)

    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.20),
        layers.RandomContrast(0.15),
    ], name="aug")

    def prep_train(img, lbl):
        img = aug(img, training=True)
        img = tf.cast(img, tf.float32) / 255.0
        return img, lbl

    def prep_val(img, lbl):
        img = tf.cast(img, tf.float32) / 255.0
        return img, lbl

    train_ds = train_ds.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(prep_val,   num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_weights


# ---- model ----

def build_model() -> keras.Model:
    """
    MobileNetV2 backbone + classification head.
    Base starts fully frozen — call prepare_phase2() before fine-tuning.

    Rescaling [0,1] -> [-1,1] is baked in so inference preprocessing stays unchanged.
    training=False on the base keeps BatchNorm using stored ImageNet stats in both phases,
    which is the standard approach for MobileNetV2 transfer learning.
    """
    inputs = layers.Input((*IMG_SIZE, 3), name="input")
    x      = layers.Rescaling(scale=2.0, offset=-1.0, name="rescale")(inputs)

    base = keras.applications.MobileNetV2(
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
        weights="imagenet",
    )
    base.trainable = False  # frozen for phase 1; unfrozen in prepare_phase2()

    x   = base(x, training=False)   # always use stored BN stats
    x   = layers.GlobalAveragePooling2D(name="gap")(x)
    x   = layers.Dropout(0.30)(x)
    x   = layers.Dense(256, activation="relu", name="fc1")(x)
    x   = layers.Dropout(0.20)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    return models.Model(inputs, out, name="glasses_mobilenetv2")


def prepare_phase2(model: keras.Model) -> None:
    """
    Unfreeze the top layers of MobileNetV2 in-place for phase 2.
    Modifies the existing model rather than rebuilding — avoids weight shape mismatches
    that occur when trying to load_weights() onto a freshly instantiated MobileNetV2.
    """
    for layer in model.layers:
        if isinstance(layer, keras.Model):       # the MobileNetV2 sub-model
            layer.trainable = True
            for sub_layer in layer.layers[:FINE_TUNE_FROM]:
                sub_layer.trainable = False      # keep early feature layers frozen
            n_trainable = sum(1 for l in layer.layers if l.trainable)
            print(f"  MobileNetV2: {n_trainable}/{len(layer.layers)} layers now trainable")
            return


def make_callbacks() -> list:
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-8, verbose=1
        ),
        callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
    ]


# ---- training ----

def main():
    print("=" * 55)
    print("  Glasses Detection — Transfer Learning (MobileNetV2)")
    print("=" * 55)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"\nDataset not found at {DATASET_DIR}\n"
            "Run build_dataset.py first, or place glasses/ and no_glasses/ folders there.\n"
            "For best results use MeGlass: https://tinyurl.com/meglass-dataset"
        )

    train_ds, val_ds, class_weights = load_datasets()

    model = build_model()
    total = sum(np.prod(v.shape) for v in model.variables)

    # ------------------------------------------------------------------ Phase 1
    print("\n" + "=" * 55)
    print("  Phase 1 — Head only (MobileNetV2 fully frozen)")
    print("=" * 55)

    trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"Trainable params: {trainable:,} / {total:,}\n")

    model.compile(
        optimizer=keras.optimizers.Adam(PHASE1_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=make_callbacks(),
        class_weight=class_weights,
        verbose=1,
    )

    # ------------------------------------------------------------------ Phase 2
    print("\n" + "=" * 55)
    print(f"  Phase 2 — Fine-tuning top layers (from {FINE_TUNE_FROM})")
    print("=" * 55)

    # Modify the existing model in-place — no rebuild, no load_weights, no shape errors
    prepare_phase2(model)

    trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"Trainable params: {trainable:,} / {total:,}\n")

    model.compile(
        optimizer=keras.optimizers.Adam(PHASE2_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    h = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=make_callbacks(),
        class_weight=class_weights,
        verbose=1,
    )

    best_acc = max(h.history["val_accuracy"])
    best_auc = max(h.history["val_auc"])
    print(f"\nBest val accuracy : {best_acc:.4f}")
    print(f"Best val AUC      : {best_auc:.4f}")
    print(f"Model saved       -> {MODEL_PATH}")
    print("\nNOTE: glasses_model.py IMG_SIZE must be (96, 96)")
    print("      Better data: https://tinyurl.com/meglass-dataset")


if __name__ == "__main__":
    main()