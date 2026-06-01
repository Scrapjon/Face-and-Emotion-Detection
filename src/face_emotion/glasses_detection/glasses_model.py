import warnings
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# alphabetical keras ordering: glasses=0, no_glasses=1
# sigmoid >= 0.5 -> class 1 (no glasses)
# sigmoid <  0.5 -> class 0 (glasses detected)

# IMPORTANT: must match IMG_SIZE in train_glasses.py
IMG_SIZE = (96, 96)


class GlassesDetector:
    def __init__(self, model_path: Path | str = None):
        self.model = None
        if model_path is not None:
            model_path = Path(model_path)
        else:
            current_dir = Path(__file__).resolve().parent
            target_path = Path("src", "models", "glasses_model.h5")
            anchor = current_dir

            for _ in range(4):
                if (anchor / target_path).exists():
                    model_path = anchor / target_path
                    break
                if (anchor / "models" / "glasses_model.h5").exists():
                    model_path = anchor / "models" / "glasses_model.h5"
                    break
                anchor = anchor.parent

            if model_path is None or not model_path.exists():
                model_path = Path("src", "models", "glasses_model.h5")

        print(f"\n[GlassesDetector] TARGET PATH RESOLVED TO:\n -> {model_path.resolve()}")
        print(f"[GlassesDetector] FILE EXISTS: {model_path.exists()}\n")

        if not model_path.exists():
            warnings.warn(f"[GlassesDetector] No model found at {model_path.resolve()}. Detector disabled.")
            return

        try:
            self.model = tf.keras.models.load_model(str(model_path), compile=False)
            print(f"[GlassesDetector] SUCCESS: Model loaded into memory!")
        except Exception as e:
            warnings.warn(f"[GlassesDetector] CRASH DURING LOAD: {e}")

    def is_available(self) -> bool:
        return self.model is not None

    def preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_bgr, IMG_SIZE)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = rgb.astype("float32") / 255.0   # [0,1]; model's Rescaling layer handles [-1,1]
        return np.expand_dims(tensor, axis=0)

    def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
        """
        Returns ('Glasses', confidence) or ('No Glasses', confidence).
        Falls back to ('Unknown', 0.0) if the model isn't loaded or the crop is bad.
        """
        if self.model is None or face_bgr is None or face_bgr.size == 0:
            return ("Unknown", 0.0)

        try:
            tensor     = self.preprocess_face(face_bgr)
            prediction = float(self.model.predict(tensor, verbose=0)[0][0])

            # sigmoid output: class 1 (no glasses) when >= 0.5, class 0 (glasses) when < 0.5
            if prediction >= 0.5:
                return ("No Glasses", prediction)
            else:
                return ("Glasses", 1.0 - prediction)

        except Exception as e:
            print(f"[GlassesDetector] predict failed: {e}")
            return ("Unknown", 0.0)


if __name__ == "__main__":
    detector = GlassesDetector()
    print(f"Available: {detector.is_available()}")
    if detector.is_available():
        fake = np.zeros((80, 80, 3), dtype=np.uint8)
        label, conf = detector.predict(fake)
        print(f"Test: {label} ({conf:.3f})")