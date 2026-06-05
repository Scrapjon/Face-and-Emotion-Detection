# %%
# -- Dependencies --

import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D,
                                     AveragePooling2D,
                                     MaxPooling2D,
                                     Flatten,
                                     Dense,
                                     Dropout,
                                     Rescaling)

# %%
# -- Base model --

num_classes = 7 # angry, disgust, fear, happy, neutral, sad, surprise

def base_model():
    model = Sequential()

    model.add(Conv2D(64, (5,5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2,2)))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    return model

# %%
# -- Load pretrained weights --

WEIGHTS_URL = 'https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5'

# Download file to '~/.keras/models/'
weights_path = tf.keras.utils.get_file('facial_expression_model_weights.h5', origin=WEIGHTS_URL, cache_subdir='models')

model = base_model()
model.load_weights(weights_path)
model.summary()

# %%
# -- Preparing dataset --

# Get current script directory
script_dir = os.path.dirname(__file__)

data_dir = os.path.abspath(os.path.join(script_dir, "../../data/emotion_data")) # absolute path makes path clean (removes ../../)
train_dir = data_dir + '/train'

# Load datasets
from tensorflow.keras.utils import image_dataset_from_directory

BATCH_SIZE = 32
IMG_SIZE = (48, 48)
SEED = 123

train_ds = image_dataset_from_directory(
    train_dir,
    label_mode='categorical', # Coverts labels to vector for layer.Dense(num_classes)
    color_mode='grayscale', # Otherwise it will assume 3 colour channels even for grayscale input
    validation_split=0.2,
    subset='training',
    seed=SEED,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

val_ds = image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    color_mode='grayscale',
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# Normalise inputs
rescale = Rescaling(1./255) # Keras layer, scaled 0-1 bc base model trained on 0-1 inputs

train_ds = train_ds.map(lambda x, y: (rescale(x), y))
val_ds = val_ds.map(lambda x, y: (rescale(x), y))

# %%
# -- Fine-tuning --

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Create path for fine-tuned weights
checkpoint_dir = os.path.join(script_dir, 'fine_tuned_weights')
# Create folder if it does not exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir, 'emotion_model_best.weights.h5')

# Checkpoint incase crash
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True, # Overwrite wih best model version
    save_freq='epoch'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Freeze layers (prevent overfitting + pretrained weights are already good)
for layer in model.layers[:8]:
    layer.trainable=False

# Compile model
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy', # Alr built into keras internal dict # from_logits=False default 
    metrics=['accuracy'] # Keras assumes accuracy metric based on loss function
)

# %%
# -- Train model --

num_epochs = 5

history = model.fit(
    train_ds,
    epochs=num_epochs,
    validation_data=val_ds,
    callbacks=[checkpoint, early_stopping]
)

# -- Visualisation --
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Category Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# %%
# -- Save fine-tuned model --

# Create path for fine-tuned model
ft_model_dir = os.path.join(script_dir, 'fine_tuned_models')
# Create folder if it does not exist
if not os.path.exists(ft_model_dir):
    os.makedirs(ft_model_dir)

model.save(os.path.join(ft_model_dir, 'ft_emotion_model.h5'))

# %%
# -- Confusion matrix --

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 1. Extract class names from the dataset
class_names = sorted([d for d in os.listdir(train_dir) 
if os.path.isdir(os.path.join(train_dir, d))])

# 2. Collect true labels and predictions
y_true = []
y_pred = []

print("Generating predictions for confusion matrix...")
for images, labels in val_ds:
    # Get the actual labels (convert from one-hot to integer)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    # Get the model predictions
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

# 3. Create the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# 4. Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 5. Print a text report (Precision, Recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
# %%
