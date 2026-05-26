import cv2 as cv
import tkinter as tk
import sys
from pathlib import Path
from typing import Dict, Tuple
from cv2.typing import MatLike
import numpy as np
from PIL import Image, ImageTk
from threading import Thread
from face_emotion.face_recognition import FacialRecognitionModel
# Constants
CAMERA: Dict[str, int] = { # might be different for ur computer
	'BACK'	: 0,
	'FRONT'	: 1
}
WINDOW_NAME: str = 'Face and Emotion Detection'
IMAGE_SIZE = 500, 500

class App:
	db_path: Path
	model: FacialRecognitionModel
	root: tk.Tk

	frame_label: tk.Label
	current_frame = None

	def __init__(self, db_path: Path | str):
		self.model = FacialRecognitionModel(db_path)

	def detection_loop(self):
		frames = self.model.run_stream()

		for frame in frames: # infinite loop until manually broken or error
			if type(frame) != np.ndarray: continue
			frame = frame[:, :, ::-1] # Convert from BGR to RGB
			frame_image = Image.fromarray(frame)
			frame_image = frame_image.resize(IMAGE_SIZE)
			frame_image = ImageTk.PhotoImage(frame_image)
			self.frame_label.configure(image=frame_image)

	def run(self):
		self.root = tk.Tk(WINDOW_NAME)
		running = True
		self.frame_label = tk.Label(self.root)
		self.frame_label.pack(side="bottom", fill="both", expand=True)
		#self.prev_frame_label = tk.Label(self.root)
		#self.prev_frame_label.pack(side="bottom", fill="both", expand=True)

		

		detection_thread = Thread(None, self.detection_loop)
		detection_thread.start()
		self.root.mainloop()
		self.model.stop_stream()
		print("Exiting...")

# entry point
if __name__ == "__main__":

	app = App("debug_data")
	app.run()