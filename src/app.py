import cv2 as cv
import tkinter as tk
import sys
from multiprocessing import Process, Lock, Pipe
from typing import Dict, Tuple
from cv2.typing import MatLike
from PIL import Image, ImageTk
# Constants
CAMERA: Dict[str, int] = { # might be different for ur computer
	'BACK'	: 0,
	'FRONT'	: 1
}
CAM_SIZE: Tuple[int, int] = (800, 600) # width, height
WINDOW_NAME: str = 'Face and Emotion Detection'
# entry point
if __name__ == "__main__":
	#
	cap: cv.VideoCapture = cv.VideoCapture(CAMERA['FRONT'])
	cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_SIZE[0])
	cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_SIZE[1])
	running: bool = True
	#
	if not cap.isOpened():
		print('Could not open webcam')
		exit()
	# setup tk window
	app = tk.Tk()
	app.bind('<Escape>', lambda e: app.quit())
	#
	image_widget = tk.Label(app)
	image_widget.pack()
	# main loop
	while running:
		try:
			ret: bool
			cap_frame: MatLike
			ret, cap_frame = cap.read()
			cv_image: MatLike = cv.cvtColor(cap_frame, cv.COLOR_BGR2RGBA)
			capture_image = Image.fromarray(cv_image)
			photo_image = ImageTk.PhotoImage(image=capture_image)
			image_widget.photo_image = photo_image
			image_widget.configure(image = photo_image)
			# cv.imshow(WINDOW_NAME, frame)
			# if cv.waitKey(1) == ord('q') or \
			# 	cv.getWindowProperty(WINDOW_NAME, cv.WND_PROP_VISIBLE) < 1:
			# 	running = False
			
			app.mainloop()
		except Exception as e:
			line_no = sys.exc_info()[-1].tb_lineno
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = exc_tb.tb_frame.f_code.co_filename
			print(f"Error in {fname} @ line {line_no} - {e}")
	cap.release()
	#cv.destroyAllWindows()