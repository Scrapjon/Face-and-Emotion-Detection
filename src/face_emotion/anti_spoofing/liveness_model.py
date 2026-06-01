from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf
import cv2
import numpy as np

class LivenessDetector:
    

    def __init__(
        self,
        model_path: str | Path = Path("../../models", "liveness_model.h5"),
        threshold: float = 0.60,
        input_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.input_size = input_size
        self.model = None

        if self.model_path.exists():
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print(f"[LivenessDetector] No model found at {self.model_path}. Liveness check is disabled.")

    def is_available(self) -> bool:
        return self.model is not None

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """Convert an OpenCV BGR face crop to a normalized model tensor."""
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face crop supplied to liveness detector")

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, self.input_size)
        # Keep pixels in 0-255 range because the trained Keras model already
        # contains a Rescaling(1./255) layer. Dividing here as well would double-scale
        # webcam input and make predictions unreliable.
        face_rgb = face_rgb.astype("float32")
        return np.expand_dims(face_rgb, axis=0)

    def predict_live_probability(self, face_bgr: np.ndarray) -> float:
        """Return probability that the cropped face is live/real."""
        if self.model is None:
            # Do not block recognition if no liveness model has been trained yet.
            return 1.0

        tensor = self.preprocess(face_bgr)
        pred = self.model.predict(tensor, verbose=0)
        return float(pred[0][0])

    def check(self, face_bgr: np.ndarray) -> tuple[bool, float]:
        """Return (is_live, live_probability)."""
        prob = self.predict_live_probability(face_bgr)
        return prob >= self.threshold, prob
