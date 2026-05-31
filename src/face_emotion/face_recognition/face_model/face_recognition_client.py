import os
import gdown
import numpy as np
from numpy.typing import NDArray
from typing import List, cast, Any
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
class FaceRecognitionClient():
	def __init__(self):
		self.model = load_model()
		self.input_shape = (224,224)
		self.output_shape = 4096
	def forward(self, img: NDArray) -> List[float]:
		""""""
		# if not isinstance(self.model, Model):
		# 	raise ValueError("Model hasnt been loaded properly?! Something has gone wrong!")
		# if img.ndim == 3:
		# 	img = np.expand_dims(img, axis=0)
		# if img.ndim == 4 and img.shape[0] == 1:
		# 	embeddings = self.model(img, training=False).numpy()
		# if img.ndim == 4 and img.shape[0] > 1:
		# 	embeddings = self.model.predict_on_batch(img)
		# if embeddings.shape[0] == 1:
		# 	return cast(List[float], embeddings[0].tolist())
		# return cast(List[List[float]], embeddings.tolist())
		embedding = self.model.predict(img, verbose=0)[0].tolist()
		if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
			embedding = cast(List[List[float]], embedding)
			norm = np.linalg.norm(np.asarray(embedding), axis=1, keepdims=True)
			embedding_norm = cast(NDArray[Any], embedding / (norm + 1e-10))
		else:
			embedding = cast(List[float], embedding)
			norm = np.linalg.norm(np.asarray(embedding), axis=None, keepdims=True)
			embedding_norm = cast(NDArray[Any], embedding / (norm + 1e-10))
		return cast(List[float], embedding_norm.tolist())
#
def load_model():
	"""Load the model"""
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
	model: Sequential = base_model()
	def get_weight_file() -> str:
		"""Download the weights file if it isnt already downloaded"""
		file_path = "src/models/new_weights.weights.h5"
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
	# dummy forward pass to force all layer variables to be created before load_weights.
	# newer keras versions leave layers un-built until first inference, so load_weights
	# finds 0 variables per layer and throws "expected 2 variables, received 0".
	model(np.zeros((1, 224, 224, 3), dtype="float32"), training=False)
	try:
		model.load_weights(weight_file)
	except (ValueError, Exception) as e:
		print(f"[load_model] standard weight load failed ({e}), retrying with by_name=True...")
		model.load_weights(weight_file, by_name=True, skip_mismatch=True)
	model_input = model.layers[0].input
	model_output = model.layers[-1].output
	return Model(inputs=model_input, outputs=model_output)