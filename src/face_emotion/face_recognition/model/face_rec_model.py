import os
import gdown
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
	Convolution2D,
	ZeroPadding2D,
	MaxPooling2D,
	Flatten,
	Dropout,
	Activation,
)
#
WEIGHTS_URL = "https://drive.google.com/file/d/1tJ9b2xgTPfjPu8Fa-_zCpA60edzlmICU/view?usp=drive_link"
#
def base_model() -> Sequential:
	"""Create the base sequential model"""
	model = Sequential()
	model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
	model.add(Convolution2D(64, (3, 3), activation="relu"))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, (3, 3), activation="relu"))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, (3, 3), activation="relu"))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, (3, 3), activation="relu"))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, (3, 3), activation="relu"))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, (3, 3), activation="relu"))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, (3, 3), activation="relu"))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation="relu"))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation="relu"))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation="relu"))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation="relu"))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation="relu"))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, (3, 3), activation="relu"))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	# layers in base VGGFace model, not used here
	# model.add(Convolution2D(4096, (7, 7), activation="relu"))
	# model.add(Dropout(0.5))
	# model.add(Convolution2D(4096, (1, 1), activation="relu"))
	# model.add(Dropout(0.5))
	# model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation("softmax"))
	return model
#
def load_model():
	""""""
	model: Sequential = base_model()
	def get_weight_file() -> str:
		""""""
		file_path = "src/face_emotion/face_recognition/model/new_weights.weights.h5"
		target_file = os.path.normpath(file_path)
		if os.path.isfile(target_file):
			print("file exists, download not needed")
			return target_file
		try:
			print()
			gdown.download(WEIGHTS_URL, target_file, quiet=False)
		except Exception as e:
			raise ValueError(f"Something went wrong downloading the file! {e}")
		return target_file
	weight_file = get_weight_file()
	#
	model.load_weights(weight_file)
	model_input = model.layers[0].input
	model_output = model.layers[-1].output
	return Model(inputs=model_input, outputs=model_output)
