"""Prepare a liveness dataset using the Kaggle TrainingDataPro Printout dataset.

The Kaggle dataset at trainingdatapro/printout contains printed-photo attack images,
so it is used as the SPOOF class. You must still provide LIVE/REAL face images, either
from your webcam captures or another allowed real-face dataset.

Example:
    kaggle datasets download -d trainingdatapro/printout -p data/kaggle/printout --unzip
    python -m face_emotion.anti_spoofing.collect_liveness_data --label live --split raw --camera 0
    python -m face_emotion.anti_spoofing.prepare_printout_dataset \
        --spoof-dir data/kaggle/printout \
        --live-dir data/liveness_raw/live \
        --output-dir data/liveness_dataset \
        --crop-faces
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def crop_face_if_possible(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return image

    # Use the largest detected face and add a small margin.
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    margin = int(0.20 * max(w, h))
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, image.shape[1])
    y2 = min(y + h + margin, image.shape[0])
    return image[y1:y2, x1:x2]


def save_image(src: Path, dst: Path, crop_faces: bool) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if crop_faces:
        image = crop_face_if_possible(src)
        if image is None or image.size == 0:
            return False
        return cv2.imwrite(str(dst.with_suffix(".jpg")), image)

    shutil.copy2(src, dst)
    return True


def split_files(files: list[Path], val_ratio: float, max_images: int | None) -> tuple[list[Path], list[Path]]:
    rng = random.Random(SEED)
    files = files.copy()
    rng.shuffle(files)
    if max_images is not None:
        files = files[:max_images]
    split_at = max(1, int(len(files) * (1 - val_ratio)))
    return files[:split_at], files[split_at:]


def copy_split(files: list[Path], output_dir: Path, split: str, label: str, crop_faces: bool) -> int:
    saved = 0
    for idx, src in enumerate(files):
        dst = output_dir / split / label / f"{label}_{idx:05d}{src.suffix.lower()}"
        if save_image(src, dst, crop_faces):
            saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Kaggle printout spoof data plus live images for liveness training.")
    parser.add_argument("--spoof-dir", default="data/kaggle/printout", help="Folder containing the unzipped Kaggle printout dataset.")
    parser.add_argument("--live-dir", default="data/liveness_raw/live", help="Folder containing real/live face images.")
    parser.add_argument("--output-dir", default="data/liveness_dataset", help="Output folder used by train_liveness.py.")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation split ratio.")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional limit per class to balance training size.")
    parser.add_argument("--crop-faces", action="store_true", help="Crop the largest detected face before saving.")
    args = parser.parse_args()

    spoof_dir = Path(args.spoof_dir)
    live_dir = Path(args.live_dir)
    output_dir = Path(args.output_dir)

    spoof_images = find_images(spoof_dir)
    live_images = find_images(live_dir)

    if not spoof_images:
        raise FileNotFoundError(f"No spoof images found in {spoof_dir}. Download Kaggle dataset trainingdatapro/printout first.")
    if not live_images:
        raise FileNotFoundError(f"No live images found in {live_dir}. Capture live images or provide a real-face dataset.")

    max_per_class = args.max_per_class or min(len(spoof_images), len(live_images))
    spoof_train, spoof_val = split_files(spoof_images, args.val_ratio, max_per_class)
    live_train, live_val = split_files(live_images, args.val_ratio, max_per_class)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    counts = {
        "train/live": copy_split(live_train, output_dir, "train", "live", args.crop_faces),
        "val/live": copy_split(live_val, output_dir, "val", "live", args.crop_faces),
        "train/spoof": copy_split(spoof_train, output_dir, "train", "spoof", args.crop_faces),
        "val/spoof": copy_split(spoof_val, output_dir, "val", "spoof", args.crop_faces),
    }

    print("Prepared liveness dataset:")
    for key, value in counts.items():
        print(f"  {key}: {value} images")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
