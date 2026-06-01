import cv2
import numpy as np
import keras
from pathlib import Path

class GenderDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            # Get directory of gender_model.py
            current_dir = Path(__file__).resolve().parent
            # Go up to 'face_emotion' -> up to 'src' -> into 'models'
            model_path = current_dir.parent.parent / "models" / "gender_model.h5"

        try:
            self.model = keras.models.load_model(str(model_path), compile=False)
            print(f"[GenderDetector] Successfully loaded from {model_path}")
        except Exception as e:
            print(f"[GenderDetector] Error: {e}")
            self.model = None

        self.img_size = (224, 224)
        # image_dataset_from_directory sorts class names alphabetically:
        # female = 0, male = 1
        self.class_names = ["Female", "Male"]

    def is_available(self):
        # Return True if model loaded successfully
        return self.model is not None

    def preprocess_face(self, face_img):
        face_img = cv2.resize(face_img, self.img_size)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype("float32")
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def predict(self, face_img):

        # Lets model run even if ft_race_model fails
        if not self.is_available():
            return 'Unknown', 0.0 # label = Unknown, confidence = 0.0
        
        processed = self.preprocess_face(face_img)
        prediction = self.model.predict(processed, verbose=0)[0][0]

        if prediction >= 0.5:
            label = "Male"
            confidence = prediction
        else:
            label = "Female"
            confidence = 1 - prediction

        return label, float(confidence)
    
if __name__ == "__main__":
    detector = GenderDetector()
    print(f"Available: {detector.is_available()}")
    if detector.is_available():
        fake = np.zeros((224, 224, 3), dtype=np.uint8)
        label, conf = detector.predict(fake)
        print(f"Test: {label} ({conf:.3f})")