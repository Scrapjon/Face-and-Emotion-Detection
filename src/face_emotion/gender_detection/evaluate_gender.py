import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = "../../data/raw/gender/Validation"
MODEL_PATH = "../../models/gender_model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

model = tf.keras.models.load_model(MODEL_PATH)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = val_ds.class_names
print("Class names:", class_names)

y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)

    predicted_labels = (predictions >= 0.5).astype(int).flatten()

    y_true.extend(labels.numpy().astype(int).flatten())
    y_pred.extend(predicted_labels)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))