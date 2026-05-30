from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import pandas as pd

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def ensure_dirs(output_dir: Path) -> None:
    for split in ["train", "val"]:
        for label in ["live", "spoof"]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)


def clear_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dirs(output_dir)


def largest_face_crop(frame):
    """Crop the largest detected face. If no face is found, return the full frame."""
    if frame is None or frame.size == 0:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

    if len(faces) == 0:
        return frame

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    # Add a small margin so the model can learn screen/print boundary clues near the face.
    margin = int(0.18 * max(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    return frame[y1:y2, x1:x2]


def save_frame(frame, output_path: Path, img_size: int) -> bool:
    crop = largest_face_crop(frame)
    if crop is None or crop.size == 0:
        return False

    crop = cv2.resize(crop, (img_size, img_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), crop))


def copy_image(raw_root: Path, rel_path: str, output_dir: Path, label: str, index: int, img_size: int) -> int:
    image_path = raw_root / rel_path
    if not image_path.exists():
        print(f"[WARN] Missing image: {image_path}")
        return 0

    frame = cv2.imread(str(image_path))
    output_path = output_dir / label / f"{label}_selfie_{index:05d}.jpg"
    return 1 if save_frame(frame, output_path, img_size) else 0


def extract_video_frames(
    raw_root: Path,
    rel_path: str,
    output_dir: Path,
    label: str,
    index: int,
    img_size: int,
    frame_step: int,
    max_frames_per_video: int,
) -> int:
    video_path = raw_root / rel_path
    if not video_path.exists():
        print(f"[WARN] Missing video: {video_path}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return 0

    saved = 0
    frame_number = 0
    while saved < max_frames_per_video:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_number % frame_step == 0:
            output_path = output_dir / label / f"{label}_video_{index:05d}_{saved:03d}.jpg"
            if save_frame(frame, output_path, img_size):
                saved += 1

        frame_number += 1

    cap.release()
    return saved


def split_indices(num_rows: int, val_ratio: float, seed: int) -> tuple[set[int], set[int]]:
    indices = list(range(num_rows))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(num_rows * val_ratio))
    val_indices = set(indices[:val_count])
    train_indices = set(indices[val_count:])
    return train_indices, val_indices


def prepare_dataset(
    raw_root: Path,
    csv_path: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    frame_step: int,
    max_frames_per_video: int,
    img_size: int,
    clean: bool,
) -> None:
    if clean:
        clear_output(output_dir)
    else:
        ensure_dirs(output_dir)

    df = pd.read_csv(csv_path)
    required_columns = {"live_selfie", "live_video", "attack"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    train_indices, val_indices = split_indices(len(df), val_ratio, seed)

    counts = {"train_live": 0, "train_spoof": 0, "val_live": 0, "val_spoof": 0}

    for idx, row in df.iterrows():
        split = "val" if idx in val_indices else "train"
        split_dir = output_dir / split

        # LIVE: one selfie image + sampled live-video frames.
        counts[f"{split}_live"] += copy_image(
            raw_root, str(row["live_selfie"]), split_dir, "live", idx, img_size
        )
        counts[f"{split}_live"] += extract_video_frames(
            raw_root,
            str(row["live_video"]),
            split_dir,
            "live",
            idx,
            img_size,
            frame_step,
            max_frames_per_video,
        )

        # SPOOF: sampled attack-video frames.
        counts[f"{split}_spoof"] += extract_video_frames(
            raw_root,
            str(row["attack"]),
            split_dir,
            "spoof",
            idx,
            img_size,
            frame_step,
            max_frames_per_video,
        )

    print("\nPrepared liveness dataset:")
    for key, value in counts.items():
        print(f"  {key}: {value} images")
    print(f"\nOutput folder: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Kaggle Printout CSV dataset for liveness training.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/printout"),
        help="Folder containing printed_photos.csv, live_selfie, live_video and attack.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to printed_photos.csv. Defaults to <raw-root>/printed_photos.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/liveness_dataset"),
        help="Prepared output dataset folder.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-step", type=int, default=10, help="Use every Nth video frame.")
    parser.add_argument("--max-frames-per-video", type=int, default=25)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--no-clean", action="store_true", help="Do not delete existing prepared dataset first.")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv if args.csv is not None else args.raw_root / "printed_photos.csv"
    prepare_dataset(
        raw_root=args.raw_root,
        csv_path=csv_path,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        frame_step=args.frame_step,
        max_frames_per_video=args.max_frames_per_video,
        img_size=args.img_size,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
