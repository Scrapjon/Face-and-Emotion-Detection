import cv2
import numpy as np
import tensorflow as tf


class GenderDetector:
    def __init__(self, model_path="../../models/gender_model.h5"):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (224, 224)

        # image_dataset_from_directory sorts class names alphabetically:
        # female = 0, male = 1
        self.class_names = ["Female", "Male"]

    def preprocess_face(self, face_img):
        face_img = cv2.resize(face_img, self.img_size)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype("float32")
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def predict(self, face_img):
        processed = self.preprocess_face(face_img)

        prediction = self.model.predict(processed, verbose=0)[0][0]

        if prediction >= 0.5:
            label = "Male"
            confidence = prediction
        else:
            label = "Female"
            confidence = 1 - prediction

        return label, float(confidence)