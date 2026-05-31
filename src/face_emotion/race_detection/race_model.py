# %%
# -- Dependencies --

import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (ZeroPadding2D,
                                     Conv2D,
                                     MaxPooling2D,
                                     Flatten,
                                     Dense,
                                     Dropout,
                                     Rescaling,
                                     Activation)

# %%
# -- Base model --

num_classes = 5 # Asian, Black, Indian, White, Other (Latino or Middle Eastern) 

def base_model() -> Sequential:
  model = Sequential()
  model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
  model.add(Conv2D(64, (3, 3), activation="relu"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(64, (3, 3), activation="relu"))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(128, (3, 3), activation="relu"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(128, (3, 3), activation="relu"))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(256, (3, 3), activation="relu"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(256, (3, 3), activation="relu"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(256, (3, 3), activation="relu"))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(512, (3, 3), activation="relu"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(512, (3, 3), activation="relu"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(512, (3, 3), activation="relu"))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(512, (3, 3), activation="relu"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(512, (3, 3), activation="relu"))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(512, (3, 3), activation="relu"))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  
  model.add(Conv2D(4096, (7, 7), activation="relu"))
  model.add(Dropout(0.5))
  model.add(Conv2D(4096, (1, 1), activation="relu"))
  model.add(Dropout(0.5))
  model.add(Conv2D(2622, (1, 1)))
  model.add(Flatten())
  model.add(Activation('softmax'))

  return model

# %%
# -- Load pretrained weights --

WEIGHTS_URL = 'https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5'

# Download file to '~/.keras/models/'
weights_path = tf.keras.utils.get_file('vgg_face_weights.h5', origin=WEIGHTS_URL, cache_subdir='models')

model = base_model()
model.load_weights(weights_path)
model.summary()

# %%
# -- Transfer learning --

# Modify architecture for 5 outputs (Top-Swap)
model.pop() # Remove Activation
model.pop() # Remove Flatten
model.pop() # Remove Conv2D(2622)

model.add(Conv2D(num_classes, (1, 1), name='predictions'))
model.add(Flatten())
model.add(Activation('softmax'))

model.summary()

# %%
# -- Preparing dataset --

import shutil

script_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(script_dir, "../../data/race_data"))

'''
CREATING TRAIN FOLDER FROM RAW DATA. NO LONGER NEEDED AFTER FIRST RUN:

# Path to raw UTKFace cropped folder (labels part of filename: [age]_[gender]_[race]_[fate&time].jpg)
raw_data_dir = data_dir + '/raw_train'
# Create file for sorted labels (new train dataset)
output_dir = data_dir + '/train'

race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

# Create the folders
for name in race_map.values():
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)

# Move the files
for filename in os.listdir(raw_data_dir):
    parts = filename.split('_')
    if len(parts) > 2:
        try:
            race_idx = int(parts[2])
            if race_idx in race_map:
                source = os.path.join(raw_data_dir, filename)
                destination = os.path.join(output_dir, race_map[race_idx], filename)
                shutil.copy(source, destination) # or shutil.move
        except: continue
        
THEREFORE RAW DATA DELETED, ONLY TRAIN FOLDER:
'data/race_data/raw_data/filename.jpg' --> 'data/race_data/train/White/filename.jpg'
                                                                /Black/..
                                                                /Asian/..
                                                                /Indian/..
                                                                /Others/..
'''

train_dir = data_dir + '/train'

# Load datasets
from tensorflow.keras.utils import image_dataset_from_directory

BATCH_SIZE = 32
IMG_SIZE = (224, 224) # VGG-Face requirement
SEED = 123

train_ds = image_dataset_from_directory(
    train_dir,
    label_mode='categorical', # Coverts labels to vector for layer.Dense(num_classes)
    validation_split=0.2,
    subset='training',
    seed=SEED,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

val_ds = image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# Normalise inputs
rescale = Rescaling(1./255) # Base model trained on 0-1 inputs

train_ds = train_ds.map(lambda x, y: (rescale(x), y))
val_ds = val_ds.map(lambda x, y: (rescale(x), y))

# %%
# -- Fine-tuning --

# Create class weights
# 0: Asian, 1: Black, 2: Indian, 3: Others, 4: White
counts = [1553, 405, 1452, 1103, 5265] # label counts already known (not ideal for real implementation, I know)
total_samples = sum(counts)

# Calculate weights
class_weights = {}
for i, count in enumerate(counts):
    weight = total_samples / (num_classes * count)
    class_weights[i] = weight

print("Calculated Class Weights:", class_weights)

# Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

checkpoint_dir = os.path.join(script_dir, 'fine_tuned_weights')
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir, 'race_model_best.weights.h5')

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True,
    save_freq='epoch'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Freeze layers (keep VGG feature extractor locked)
for layer in model.layers[:-7]:
    layer.trainable = False

# Compile model
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %%
# -- Train model --

num_epochs = 1

history = model.fit(
    train_ds,
    epochs=num_epochs,
    validation_data=val_ds,
    callbacks=[checkpoint, early_stopping],
    class_weight=class_weights
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
plt.title('Training and Validation Accuracy (Race)')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss (Race)')
plt.show()

# %%
# -- Save fine-tuned model --

ft_model_dir = os.path.join(script_dir, 'fine_tuned_models')
if not os.path.exists(ft_model_dir): os.makedirs(ft_model_dir)

model.save(os.path.join(ft_model_dir, 'ft_race_model.h5'))