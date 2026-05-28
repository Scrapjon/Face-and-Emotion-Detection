"""Collect live/spoof face samples using your webcam.

Examples:
python -m face_emotion.anti_spoofing.collect_liveness_data --label live --split train
python -m face_emotion.anti_spoofing.collect_liveness_data --label spoof --split train
python -m face_emotion.anti_spoofing.collect_liveness_data --label live --split val
python -m face_emotion.anti_spoofing.collect_liveness_data --label spoof --split val

For spoof samples, show printed photos or phone-screen face images to the camera.
Press SPACE to save the current detected face crop. Press ESC to quit.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["live", "spoof"], required=True)
    parser.add_argument("--split", choices=["train", "val", "raw"], default="train")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", default="data/liveness_dataset", help="Output root. Use --out data/liveness_raw with --split raw when preparing a Kaggle-based dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out) / args.split / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try --camera 1 if needed.")

    print("SPACE = save face crop | ESC = quit")
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        face_crop = None
        for (x, y, w, h) in faces[:1]:
            pad = int(0.15 * max(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            face_crop = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"{args.split}/{args.label} saved={saved}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Collect liveness data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        if key == 32 and face_crop is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = out_dir / f"{args.label}_{ts}.jpg"
            cv2.imwrite(str(path), face_crop)
            saved += 1
            print("Saved", path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
