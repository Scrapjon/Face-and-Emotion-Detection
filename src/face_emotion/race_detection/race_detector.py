import os
import gdown
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

class RaceDetector:
    def __init__(self, model_path=None):
        self.model = None # Define as None for thread safety

        # file id from google.drive link
        self.file_id = '1uJqSEqgH3ovZk0pFU3CloGl54Kg4xZj7'
        self.class_names = ['Asian', 'Black', 'Indian', 'Others', 'White']
        self.img_size = (224, 224)
        self.model = None
        
        if model_path is None:
            # Get directory of THIS file (src/face_emotion/race_detection/)
            current_dir = Path(__file__).resolve().parent
            
            # Go up two levels to get to 'src', then into 'models'
            # (race_detection -> face_emotion -> src -> models)
            model_dir = current_dir.parent.parent / 'models'
            model_path = model_dir / 'ft_race_model.h5'

            # Download if does not already exist
            if not model_path.exists():
                print('[RaceDetector] Downloading 1.4GB model to {model_path}...')
                url = f'https://drive.google.com/uc?id={self.file_id}'
                gdown.download(url, str(model_path), quiet=False)

        try:
            self.model = tf.keras.models.load_model(str(model_path), compile=False)
            print('[RaceDetector] Success: Model loaded from {model_pah}.')
        except Exception as e:
            print(f'[RaceDetector] Failed to load model: {e}')

    def is_available(self):
        # Return True if model loaded successfully
        return self.model is not None

    def preprocess_face(self, face_img):
        # Resize and convert to RGB (VGG-Face requirement)
        face_img = cv2.resize(face_img, self.img_size)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Scale to 0-1 and add batch dimension
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def predict(self, face_img):

        # Lets model run even if ft_race_model fails
        if not self.is_available():
            return 'Unknown', 0.0 # label = Unknown, confidence = 0.0
        
        processed = self.preprocess_face(face_img)
        predictions = self.model.predict(processed, verbose=0)[0] # Get all 5 probabilities
        max_index = np.argmax(predictions) # Find index of highest probability
        
        label = self.class_names[max_index]
        confidence = predictions[max_index]

        return label, float(confidence)
    
if __name__ == '__main__':
    # check if model loads and runs a fake inference
    detector = RaceDetector()
    print(f'RaceDetector available: {detector.is_available()}')
    if detector.is_available():
        # fake face crop to make sure the pipeline doesnt crash
        fake_face = np.zeros((224, 224, 3), dtype=np.uint8)
        label, probs = detector.predict(fake_face)
        print(f'Test label: {label}')
        print(f'Test probs: {probs}')