from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple
import keras

import cv2
import numpy as np

# Alphabetical order - matches how keras ImageDataGenerator.flow_from_directory() sorts subfolders
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_INPUT_SIZE = (48, 48)
DEFAULT_MODEL_PATH = Path("src/face_emotion/emotion_recognition/fine_tuned_models/ft_emotion_model.h5")


class EmotionDetector:
    # Loads the fine-tuned emotion CNN and runs inference on face crops

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model = None
        model_path = Path(model_path)

        if not model_path.exists():
            warnings.warn(f"[EmotionDetector] No model found at {model_path}. Emotion detection is disabled.")
            return

        try:
            self.model = keras.models.load_model(str(model_path))
            print(f"[EmotionDetector] Loaded model from {model_path}")
        except Exception as e:
            warnings.warn(f"[EmotionDetector] Failed to load model: {e}")

    def is_available(self) -> bool:
        # Returns True if model loads successfully
        return self.model is not None

    def detect(self, face_bgr: np.ndarray) -> Tuple[str, dict]:
        """
        detect emotion from a BGR face crop (straight from opencv).
        returns (label, probs_dict) where label is the top-1 class name and
        probs_dict maps all 7 class names to their softmax scores.
        returns ('unknown', {}) if model isnt loaded or the crop is garbage.
        """
        if self.model is None or face_bgr is None or face_bgr.size == 0:
            return ('unknown', {})

        try:
            # convert to grayscale, resize to 48x48, normalise to 0-1
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, EMOTION_INPUT_SIZE)
            normalized = resized.astype('float32') / 255.0
            # add batch dim + channel dim: (48, 48) -> (1, 48, 48, 1)
            tensor = normalized.reshape(1, 48, 48, 1)

            preds = self.model.predict(tensor, verbose=0)[0]  # shape: (7,)
            label = EMOTION_LABELS[int(np.argmax(preds))]
            probs = {k: float(v) for k, v in zip(EMOTION_LABELS, preds)}
            return (label, probs)

        except Exception as e:
            print(f"[EmotionDetector] predict blew up: {e}")
            return ('unknown', {})


if __name__ == "__main__":
    # quick sanity check - loads model and runs a fake inference
    detector = EmotionDetector()
    print(f"EmotionDetector available: {detector.is_available()}")
    if detector.is_available():
        # fake face crop to make sure the pipeline doesnt crash
        fake_face = np.zeros((80, 80, 3), dtype=np.uint8)
        label, probs = detector.detect(fake_face)
        print(f"Test label: {label}")
        print(f"Test probs: {probs}")