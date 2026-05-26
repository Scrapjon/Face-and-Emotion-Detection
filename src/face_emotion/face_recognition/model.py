import os
from pathlib import Path
import time
from threading import Thread
import cv2
import pandas as pd
from deepface import DeepFace
from dataclasses import dataclass

CAMERA_FPS_CAP = 30

UNKNOWN_NAME = "WHO ARE YOU???"
WINDOW_TITLE = "IMAGE RECOGNITION WOAHHHHH!"


# ahh maybe needs a better name lol
class FacialRecognitionModel:
    MODEL_NAME = "Facenet"
    DETECTOR = "opencv"
    DISTANCE_METRIC = "cosine"

    prev_recognitions = []
    detection_thread: Thread | None = None

    def __init__(self, db_path: Path | str = Path("data", "classification_data", "train_data")) -> None:
        self.db_path = db_path
    
    def detect(self, frame):
        """
        detects faces (duh!)
        Args:
            frame (dunno): the frame that faces should be in hopefully. also should be BGR formatting (not RGB)
        THATS ALL THE ARGS!
        """
        results = []
        try:
            # dfs does not stand for deepfaces it stands for dataframes. its confusing ik.
            dfs = DeepFace.find(
                img_path=frame,
                db_path=str(self.db_path),
                model_name=self.MODEL_NAME,
                detector_backend=self.DETECTOR,
                distance_metric=self.DISTANCE_METRIC,
                enforce_detection=False,
                silent=True
            )
        except: # either no face in frame or no images in DB
            print("Nothing seen")
            return results
        
        # get bounding box data
        for df in dfs:
            df = pd.DataFrame(df) # shut up type checker!
            try:
                face_data = df.iloc[0]
                x = int(face_data['source_x'])
                y = int(face_data['source_y'])
                w = int(face_data['source_w'])
                h = int(face_data['source_h'])
                
            except (KeyError, IndexError):
                x = y = w = h = 0
            if len(df) == 0:
                name = UNKNOWN_NAME
            else:
                identity_path = df.iloc[0]['identity'] # face_data may be unbound so just grab it again
                """
                pretty sick that you can get the closest image to you. (im adding that as a cool feature)
                we can display next to the webcam the image it is the most confident you resemble.
                for example: Oliver would look a LOT like Chris Hemsworth so put an image of shirtless Chris Hemsworth on screen.
                """
                name = os.path.basename(os.path.dirname(identity_path))

            results.append({'name': name, 'box': (x, y, w, h)})
        return results
    
    @staticmethod
    def draw_boxes(frame, recognitions):
        for r in recognitions:
            x, y, w, h = r['box']
            name = r['name']
            color = (0, 255, 0) if name != UNKNOWN_NAME else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(frame, name, (x, label_y), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
        return frame
    
    def detect_and_assign(self, frame):
        self.prev_recognitions = self.detect(frame)
    
    def async_detect(self, frame):

        # Filter out NoneType
        if self.detection_thread is None:
            self.detection_thread = Thread(None, self.detect_and_assign, args=(frame,))
            self.detection_thread.start()

        # Check process isn't overlapping
        elif not self.detection_thread.is_alive():
            self.detection_thread = Thread(None, self.detect_and_assign, args=(frame,))
            self.detection_thread.start()
        
        return self.draw_boxes(frame, self.prev_recognitions)
    
    def run_stream(self, camera_index = 0, standalone = False):
        """
        Runs the video stream which is directly connected to the deepface model
        Args:
            camera_index (int): The index of the camera on the machine that the program will capture video from.
            standalone (bool): Determines whether the stream will run as its own application or as a frame generator. Set to False if plugging into ui such as TKinter and True for running standalone. \
                You need to unpack the function like a generator though to use standalone. \
                If using standalone = True, its recommended to use the function FacialRecognitionModel.run_standalone()
        """
        self.prev_recognitions = []
        self.detection_active = True
        if standalone:
            cv2.namedWindow(WINDOW_TITLE)
        vc = cv2.VideoCapture(camera_index)

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
            raise RuntimeError("NO CAMERA AAAAAAAAH (maybe try a different camera_index value...)")
        
        try:
            while rval:
                rval, frame = vc.read()

                if standalone:
                    key = cv2.waitKey(20)
                    if key == 27: # exit on ESC
                        break
                    if key == 100: # toggle detection on D
                        self.detection_active = not self.detection_active

                if self.detection_active:
                    display_frame = self.async_detect(frame)
                else:
                    display_frame = frame

                if standalone:
                    cv2.imshow(WINDOW_TITLE, display_frame)
                else:
                    yield display_frame, 
        finally:
            vc.release()
            if standalone:
                cv2.destroyAllWindows()
    def run_standalone(self, camera_index = 0):
        stream = self.run_stream(camera_index, True)
        next(stream)
if __name__ == "__main__":
    model = FacialRecognitionModel(Path("debug_data"))
    model.run_standalone()