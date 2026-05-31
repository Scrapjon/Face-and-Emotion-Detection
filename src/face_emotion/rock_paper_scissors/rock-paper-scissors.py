import tensorflow as tf
class rock_paper_scissors():
	def __init__(self):
		self.model_path = "src/face_emotion/rock_paper_scissors/model/"#rock-paper-scissors-model.pb"
		self.model = tf.keras.layers.TFSMLayer(self.model_path, call_endpoint='serving_default')
	def predict(self, img):
		"""
		Predict the image
		input should be 640x640
		output will be:
			0 - Paper
			1 - Rock
			2 - Scissors
		"""
		return self.model.predict(img)