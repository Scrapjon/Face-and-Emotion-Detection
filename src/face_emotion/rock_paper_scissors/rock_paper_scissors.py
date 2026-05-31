import tensorflow as tf
import cv2
import numpy as np
from typing import Optional
from ultralytics import YOLO
#
class Rock_Paper_Scissors():
	""""""
	def __init__(self, 
		path: str = "src/face_emotion/rock_paper_scissors/model/"
	):
		""""""
		self.yolo_tf = tf.saved_model.load(path)
		self.infer = self.yolo_tf.signatures["serving_default"]
	#
	def predict(self, img):
		"""
		Predict held gesture in the image
		output will be:
			0 - Paper
			1 - Rock
			2 - Scissors
		"""
		img_input = cv2.resize(img, (640, 640))
		img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
		input_tensor = np.expand_dims(img_input, axis=0).astype(np.float32)/255.0
		input_tensor = tf.convert_to_tensor(input_tensor)
		predictions = self.infer(images=input_tensor)
		raw_output = predictions['output_0'].numpy()
		def process_yolo_output(raw_output, num_classes=3, confidence_threshold=0.25, iou_threshold=0.45):
			"""
			Decodes raw YOLO tensor outputs into bounding boxes, scores, and classes.
			Supports standard transposed formats (e.g., YOLOv8 [1, 84, 8400]).
			"""
			if raw_output.shape[1] == (4 + num_classes):
					raw_output = tf.transpose(raw_output, perm=[0, 2, 1])
			
			boxes_xywh = raw_output[..., :4] 
			class_logits = raw_output[..., 4:] 
			
			cx, cy, w, h = tf.split(boxes_xywh, 4, axis=-1)
			ymin = cy - (h / 2.0)
			xmin = cx - (w / 2.0)
			ymax = cy + (h / 2.0)
			xmax = cx + (w / 2.0)
			boxes_corners = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
			
			# Compute probabilities using sigmoid
			class_probs = tf.math.sigmoid(class_logits) 
			box_scores = tf.reduce_max(class_probs, axis=-1) 
			box_classes = tf.argmax(class_probs, axis=-1) 
			boxes_for_nms = tf.expand_dims(boxes_corners, axis=2) 
			
			nms_output = tf.image.combined_non_max_suppression(
				boxes=boxes_for_nms,
				scores=class_probs,
				max_output_size_per_class=100,
				max_total_size=100,
				iou_threshold=iou_threshold,
				score_threshold=confidence_threshold,
				clip_boxes=False
			)
			return {
				"boxes": nms_output.nmsed_boxes,        # [Batch, Max_Total_Size, 4]
				"scores": nms_output.nmsed_scores,      # [Batch, Max_Total_Size]
				"classes": tf.cast(nms_output.nmsed_classes, tf.int32),  # [Batch, Max_Total_Size]
				"valid_detections": nms_output.valid_detections          # [Batch]
			}
		processed_output = process_yolo_output(raw_output)
		print(processed_output['classes'])
		return processed_output['classes']
