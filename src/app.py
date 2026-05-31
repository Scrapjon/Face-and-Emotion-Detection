import cv2 as cv
import tkinter as tk
import sys
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image, ImageTk
from threading import Thread, Lock
from face_emotion.face_recognition import FacialRecognitionModel

CAMERA: Dict[str, int] = {
    'BACK': 0,
    'FRONT': 1
}
WINDOW_NAME: str = 'Face and Emotion Detection'
IMAGE_SIZE = 500, 500


class App:
    db_path: Path
    model: FacialRecognitionModel
    root: tk.Tk
    frame_label: tk.Label

    def __init__(self, db_path: Path | str):
        self.model = FacialRecognitionModel(db_path)
        self._pending_image = None      # next frame ready to display
        self._current_photo = None      # holds reference so GC doesn't collect it
        self._lock = Lock()

    def detection_loop(self):
        for frame in self.model.run_stream():
            if not isinstance(frame, np.ndarray):
                continue
            rgb = frame[:, :, ::-1]
            img = Image.fromarray(rgb).resize(IMAGE_SIZE)
            photo = ImageTk.PhotoImage(img)
            with self._lock:
                self._pending_image = photo  # hand off to main thread

    def _poll_frame(self):
        """Called on the main thread every 16 ms (~60 fps cap). Applies any pending frame."""
        with self._lock:
            if self._pending_image is not None:
                self._current_photo = self._pending_image   # keep reference alive
                self._pending_image = None
                self.frame_label.configure(image=self._current_photo)
        self.root.after(16, self._poll_frame)

    def run(self):
        self.root = tk.Tk()
        self.root.title(WINDOW_NAME)

        self.frame_label = tk.Label(self.root)
        self.frame_label.pack(side="bottom", fill="both", expand=True)

        # Start background detection thread
        detection_thread = Thread(target=self.detection_loop, daemon=True)
        detection_thread.start()

        # Start the main-thread polling loop
        self.root.after(16, self._poll_frame)

        self.root.mainloop()
        self.model.stop_stream()
        print("Exiting...")


if __name__ == "__main__":
    app = App("debug_data")
    app.run()