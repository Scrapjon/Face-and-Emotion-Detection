
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import Sequence

DATASET_DIR = Path("data/liveness_dataset")
MODEL_OUT = Path("models/liveness_model.keras")
PLOTS_DIR = Path("models/liveness_plots")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42


class DirectoryImageSequence(Sequence):
    def __init__(
        self,
        dataset_dir: Path,
        batch_size: int,
        image_size: tuple[int, int],
        shuffle: bool = False,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.seed = seed
        self.samples = self._collect_samples()
        self.epoch = 0

    def _collect_samples(self) -> list[tuple[Path, float]]:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        live_dir = self.dataset_dir / "live"
        spoof_dir = self.dataset_dir / "spoof"
        if not live_dir.exists() or not spoof_dir.exists():
            raise FileNotFoundError(
                "Expected folders 'live' and 'spoof' under "
                f"{self.dataset_dir}. Found: {sorted(p.name for p in self.dataset_dir.iterdir() if p.is_dir())}"
            )

        samples: list[tuple[Path, float]] = []
        for label_dir, label in ((live_dir, 1.0), (spoof_dir, 0.0)):
            for path in sorted(label_dir.iterdir()):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    samples.append((path, label))

        if not samples:
            raise FileNotFoundError(f"No image files found in {self.dataset_dir}")

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(samples)

        return samples

    def _load_image(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(self.image_size)
            return np.asarray(image, dtype=np.uint8)

    def __len__(self) -> int:
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_samples = self.samples[index * self.batch_size : (index + 1) * self.batch_size]
        images = np.empty((len(batch_samples), *self.image_size, 3), dtype=np.uint8)
        labels = np.empty((len(batch_samples),), dtype=np.float32)

        for batch_index, (path, label) in enumerate(batch_samples):
            images[batch_index] = self._load_image(path)
            labels[batch_index] = label

        return images, labels

    def on_epoch_end(self) -> None:
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(self.samples)
            self.epoch += 1


def make_datasets(dataset_dir: Path):
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    train_ds = DirectoryImageSequence(
        train_dir,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_ds = DirectoryImageSequence(
        val_dir,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False,
        seed=SEED,
    )

    print("Class names:", ["live", "spoof"])
    print("IMPORTANT: live targets are encoded as 1.0 and spoof targets as 0.0.")
    return train_ds, val_ds


def build_model():
    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.08),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="live_probability")(x)

    model = models.Model(inputs, outputs, name="custom_liveness_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for metric in ["accuracy", "loss"]:
        if metric not in history.history:
            continue
        plt.figure()
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(f"Liveness model {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{metric}.png")
        plt.close()


def main():
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = make_datasets(DATASET_DIR)
    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_history(history)
    model.save(MODEL_OUT)
    print(f"Saved liveness model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
