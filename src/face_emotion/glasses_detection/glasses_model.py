import warnings
import cv2
import numpy as np
from pathlib import Path


# alphabetical keras ordering: glasses=0, no_glasses=1
# sigmoid >= 0.5 -> class 1 (no glasses)
# sigmoid <  0.5 -> class 0 (glasses detected)
IMG_SIZE = (64, 64)  # what the model was trained on


class GlassesDetector:
    def __init__(self, model_path: Path | str = Path("src/models/glasses_model.h5")):
        self.model = None
        model_path = Path(model_path)

        if not model_path.exists():
            warnings.warn(f"[GlassesDetector] No model at {model_path}. Glasses detection disabled.")
            return

        try:
            import keras
            self.model = keras.models.load_model(str(model_path), compile=False)
            print(f"[GlassesDetector] Loaded model from {model_path}")
        except Exception as e:
            warnings.warn(f"[GlassesDetector] Failed to load model: {e}")

    def is_available(self) -> bool:
        return self.model is not None

    def preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        resized  = cv2.resize(face_bgr, IMG_SIZE)
        rgb      = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor   = rgb.astype("float32") / 255.0
        return np.expand_dims(tensor, axis=0)

    def predict(self, face_bgr: np.ndarray) -> tuple[str, float]:
        """
        returns ('Glasses', confidence) or ('No Glasses', confidence).
        falls back to ('Unknown', 0.0) if the model isnt loaded or the crop is junk.
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
