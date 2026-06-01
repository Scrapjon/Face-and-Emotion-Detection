"""
build_glasses_dataset.py
------------------------
Sources the glasses detection training dataset by:
  1. Downloading real face images from two public GitHub repos (deepface + face_recognition)
  2. Auto-labelling them with OpenCV's eye/glasses haarcascades
  3. Generating synthetic glasses overlays to balance the positive class
  4. Cropping detected faces and saving to dataset/glasses/ and dataset/no_glasses/

Run this script from the project root:
    python src/face_emotion/glasses_detection/build_dataset.py
"""

import os
import io
import ssl
import zipfile
import urllib.request
import random
import math
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ---- config ----
OUT_DIR        = Path("data/glasses_dataset")
FACE_MIN_SIZE  = 60          # ignore faces smaller than this
IMG_SIZE       = (224, 224)  # model input size
MAX_SYNTH      = 600         # max synthetic glasses examples to generate
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---- cascade setup ----
FACE_CASCADE    = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE     = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
GLASSES_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# ---- download helpers ----

def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def download_zip_images(repo: str, branch: str = "master") -> dict[str, bytes]:
    """Download all JPEG/PNG images from a GitHub repo zip. Returns {filename: bytes}."""
    url = f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
    print(f"  Downloading {repo}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=_ctx()) as r:
        data = r.read()
    zf = zipfile.ZipFile(io.BytesIO(data))
    images = {}
    for name in zf.namelist():
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                images[name] = zf.read(name)
            except Exception:
                pass
    print(f"    -> {len(images)} images")
    return images

def bytes_to_bgr(img_bytes: bytes) -> np.ndarray | None:
    """Decode image bytes to OpenCV BGR array."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# ---- labelling ----

def detect_faces(gray: np.ndarray) -> list[tuple]:
    """Return list of (x, y, w, h) face bounding boxes."""
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE))
    return list(faces) if len(faces) > 0 else []

def classify_face_region(gray_face: np.ndarray) -> str | None:
    """
    Classify a face crop as 'glasses', 'no_glasses', or None (undecided).

    Strategy:
      - haarcascade_eye.xml detects bare eyes well but struggles through glasses
      - haarcascade_eye_tree_eyeglasses.xml detects eyes even through frames
      - If the glasses cascade finds eyes but the bare eye cascade doesn't -> glasses
      - If the bare eye cascade finds eyes clearly -> no_glasses
      - Otherwise -> None (skip)
    """
    h, w = gray_face.shape
    # scale minSize relative to face crop size
    min_eye = max(8, h // 8)

    bare_eyes   = EYE_CASCADE.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=3, minSize=(min_eye, min_eye))
    glass_eyes  = GLASSES_CASCADE.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=3, minSize=(min_eye, min_eye))

    n_bare   = len(bare_eyes)
    n_glass  = len(glass_eyes)

    if n_glass >= 2 and n_bare == 0:
        return "glasses"
    if n_glass >= 2 and n_bare == 0:
        return "glasses"
    if n_bare >= 2:
        return "no_glasses"
    if n_glass >= 1 and n_bare == 0:
        return "glasses"
    if n_bare >= 1:
        return "no_glasses"
    return None  # can't tell

# ---- synthetic glasses drawing ----

def draw_glasses(face_bgr: np.ndarray) -> np.ndarray:
    """
    Draw a simple but realistic-looking glasses shape onto a face crop.
    Uses eye cascade to find approximate eye positions so glasses land correctly.
    Falls back to proportional placement if eyes aren't detected.
    """
    img = face_bgr.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    min_eye = max(8, h // 8)
    eyes = EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(min_eye, min_eye))

    if len(eyes) >= 2:
        # sort by x to get left/right eye
        eyes = sorted(eyes, key=lambda e: e[0])
        # use top-2 by area if more than 2 detected
        eyes = sorted(eyes[:4], key=lambda e: e[2]*e[3], reverse=True)[:2]
        eyes = sorted(eyes, key=lambda e: e[0])
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]
        lx = ex1 + ew1 // 2
        ly = ey1 + eh1 // 2
        rx = ex2 + ew2 // 2
        ry = ey2 + eh2 // 2
        lens_r = max(int(max(ew1, ew2) * 0.65), 10)
    else:
        # proportional fallback: eyes are roughly at 35-40% down, 25%/75% across
        ly  = int(h * 0.38)
        ry  = ly
        lx  = int(w * 0.28)
        rx  = int(w * 0.72)
        lens_r = int(w * 0.20)

    # pick a random glasses style + colour
    style  = random.choice(["round", "rect", "oval"])
    colour = random.choice([
        (20,  20,  20 ),   # black
        (50,  30,  10 ),   # dark brown
        (60,  60, 100 ),   # dark navy
        (20,  80,  20 ),   # dark green
        (120, 80,  20 ),   # tortoiseshell brown
    ])
    thickness = random.randint(2, 4)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    def draw_lens(cx, cy, r, style):
        if style == "round":
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=colour, width=thickness)
        elif style == "rect":
            rr = int(r * 0.25)
            draw.rounded_rectangle([cx-r, cy-r, cx+r, cy+r], radius=rr, outline=colour, width=thickness)
        else:  # oval
            draw.ellipse([cx-r, cy-int(r*0.7), cx+r, cy+int(r*0.7)], outline=colour, width=thickness)

    draw_lens(lx, ly, lens_r, style)
    draw_lens(rx, ry, lens_r, style)

    # nose bridge
    bridge_y = (ly + ry) // 2
    bridge_x1 = lx + lens_r
    bridge_x2 = rx - lens_r
    if bridge_x2 > bridge_x1:
        draw.line([(bridge_x1, bridge_y), (bridge_x2, bridge_y)], fill=colour, width=thickness)

    # temples (arms of glasses going to ears)
    arm_len = int(w * 0.12)
    draw.line([(lx - lens_r, ly), (lx - lens_r - arm_len, ly - int(arm_len*0.1))], fill=colour, width=thickness)
    draw.line([(rx + lens_r, ry), (rx + lens_r + arm_len, ry - int(arm_len*0.1))], fill=colour, width=thickness)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ---- augmentation ----

def augment(img_bgr: np.ndarray, n: int = 4) -> list[np.ndarray]:
    """Return n augmented versions of img_bgr."""
    results = [img_bgr]
    h, w = img_bgr.shape[:2]

    for _ in range(n - 1):
        aug = img_bgr.copy()
        # horizontal flip
        if random.random() < 0.5:
            aug = cv2.flip(aug, 1)
        # brightness/contrast jitter
        alpha = random.uniform(0.75, 1.3)
        beta  = random.randint(-25, 25)
        aug   = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
        # small rotation
        angle = random.uniform(-12, 12)
        M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug   = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # random crop/zoom
        if random.random() < 0.4:
            margin = random.randint(5, 20)
            crop   = aug[margin:h-margin, margin:w-margin]
            aug    = cv2.resize(crop, (w, h))
        results.append(aug)
    return results

# ---- pipeline ----

def process_images(raw_images: dict[str, bytes], label_override: str | None = None) -> tuple[list, list]:
    """
    Detect faces in raw images, label them, return (face_crops, labels).
    label_override: if set, skip cascade labelling and use this label for all crops.
    """
    crops, labels = [], []
    for fname, img_bytes in raw_images.items():
        bgr = bytes_to_bgr(img_bytes)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        if not faces:
            # try the whole image as a face crop
            faces = [(0, 0, bgr.shape[1], bgr.shape[0])]
        for (x, y, w, h) in faces:
            face = bgr[max(0,y):y+h, max(0,x):x+w]
            if face.size == 0 or min(face.shape[:2]) < FACE_MIN_SIZE:
                continue
            face_resized = cv2.resize(face, IMG_SIZE)
            if label_override:
                label = label_override
            else:
                gray_face = cv2.cvtColor(cv2.resize(face, (96, 96)), cv2.COLOR_BGR2GRAY)
                label = classify_face_region(gray_face)
            if label:
                crops.append(face_resized)
                labels.append(label)
    return crops, labels

def save_dataset(crops: list, labels: list, out_dir: Path):
    """Save face crops to out_dir/glasses/ and out_dir/no_glasses/."""
    counts = {"glasses": 0, "no_glasses": 0}
    for cls in counts:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    for crop, label in zip(crops, labels):
        if label not in counts:
            continue
        idx = counts[label]
        path = out_dir / label / f"{label}_{idx:05d}.jpg"
        cv2.imwrite(str(path), crop)
        counts[label] += 1

    return counts

def main():
    print("=" * 55)
    print("  Glasses Detection Dataset Builder")
    print("=" * 55)

    # ---- 1. download real face images ----
    print("\n[1/4] Downloading face image sources...")
    sources = {
        "deepface":        ("serengil/deepface",        "master"),
        "face_recognition":("ageitgey/face_recognition","master"),
    }
    all_images: dict[str, bytes] = {}
    for name, (repo, branch) in sources.items():
        try:
            imgs = download_zip_images(repo, branch)
            # only keep images from test/example dirs (skip icons etc.)
            imgs = {k: v for k, v in imgs.items()
                    if any(d in k.lower() for d in ["test", "example", "dataset", "sample"])}
            all_images.update({f"{name}/{k}": v for k, v in imgs.items()})
            print(f"  {name}: {len(imgs)} usable images")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    print(f"  Total source images: {len(all_images)}")

    # ---- 2. detect faces and auto-label ----
    print("\n[2/4] Detecting faces and auto-labelling...")
    crops, labels = process_images(all_images)
    print(f"  Labelled face crops: {len(crops)}")
    from collections import Counter
    print(f"  Distribution: {Counter(labels)}")

    # ---- 3. balance with synthetic glasses ----
    print("\n[3/4] Generating synthetic glasses examples...")
    no_glasses_crops = [c for c, l in zip(crops, labels) if l == "no_glasses"]
    glasses_crops    = [c for c, l in zip(crops, labels) if l == "glasses"]
    target           = max(len(no_glasses_crops), len(glasses_crops), 50)
    n_synth_needed   = min(MAX_SYNTH, max(0, target - len(glasses_crops)))

    synth_base = no_glasses_crops if no_glasses_crops else crops
    if synth_base:
        synth_pool = (synth_base * ((n_synth_needed // max(len(synth_base), 1)) + 2))[:n_synth_needed]
        for face in synth_pool:
            g = draw_glasses(face)
            crops.append(g)
            labels.append("glasses")
    print(f"  Synthetic glasses added: {n_synth_needed}")
    print(f"  Final distribution: {Counter(labels)}")

    # ---- 4. augment and save ----
    print("\n[4/4] Augmenting and saving dataset...")
    aug_crops, aug_labels = [], []
    for crop, label in zip(crops, labels):
        augs = augment(crop, n=4)
        aug_crops.extend(augs)
        aug_labels.extend([label] * len(augs))

    counts = save_dataset(aug_crops, aug_labels, OUT_DIR)
    total = sum(counts.values())
    print(f"  Saved to {OUT_DIR}/")
    print(f"  glasses:    {counts['glasses']:4d} images")
    print(f"  no_glasses: {counts['no_glasses']:4d} images")
    print(f"  Total:      {total:4d} images")
    print("\nDone! Run train_glasses.py to train the model.")

if __name__ == "__main__":
    main()
