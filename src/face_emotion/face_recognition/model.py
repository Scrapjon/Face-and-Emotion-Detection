import os
from pathlib import Path
import time

import cv2
import pandas as pd
from deepface import DeepFace

CAMERA_FPS_CAP = 30
MAX_DELTA_SECONDS = (1/CAMERA_FPS_CAP) # Yeesh bad naming convention sorry

UNKNOWN_NAME = "WHO ARE YOU???"


# ahh maybe needs a better name lol
class FacialRecognitionModel:
    MODEL_NAME = "Facenet"
    DETECTOR = "opencv"
    DISTANCE_METRIC = "cosine"

    def __init__(self, db_path: Path | str = Path("data")) -> None:
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
    
    def run_stream(self, camera_index = 0):
        t1 = 0

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("NO CAMERA AAAAAAAAH (maybe try a different camera_index value...)")
        prev_recognitions = []
        frame_idx = 0
        try:
            while True:

                if (time.perf_counter() - t1) < MAX_DELTA_SECONDS: continue
                ok, frame = cap.read()
                if not ok:
                    print("failed to grab frame")
                    break
                frame_idx += 1
                
                prev_recognitions = self.detect(frame)
                display_frame = self.draw_boxes(frame, prev_recognitions)
                cv2.imshow('IMAGE RECOGNITION WOAHHHHH!', display_frame)
                t1 = time.perf_counter()
        finally:
            cap.release()
            cv2.destroyAllWindows()



                