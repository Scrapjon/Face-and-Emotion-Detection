import tensorflow as tf
import cv2
from typing import Optional
#
class Rock_Paper_Scissors():
	""""""
	def __init__(self, 
		path: Optional[str] = "src/face_emotion/rock_paper_scissors/model/"
	):
		""""""
		self.model_path = path
		self.model = tf.keras.layers.TFSMLayer(self.model_path, call_endpoint='serving_default')
	def predict(self, img):
		"""
		Predict held gesture in the image
		output will be:
			0 - Paper
			1 - Rock
			2 - Scissors
		"""
		img = cv2.resize(img, (640, 640))
		return self.model.predict(img)