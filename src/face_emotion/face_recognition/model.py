import os
from pathlib import Path
import time
from threading import Thread
from datetime import datetime
import cv2
import pandas as pd
from deepface import DeepFace
from dataclasses import dataclass
from face_emotion.anti_spoofing import LivenessDetector
from face_emotion.gender_detection.gender_model import GenderDetector

CAMERA_FPS_CAP = 30

UNKNOWN_NAME = "WHO ARE YOU???"
WINDOW_TITLE = "IMAGE RECOGNITION WOAHHHHH!"


def clear_deepface_cache():
    try:
        if hasattr(DeepFace, "clear_cache"):
            DeepFace.clear_cache()
            return
    except Exception:
        pass
    try:
        from deepface.commons import functions as deepface_functions
        if hasattr(deepface_functions, "clear_cache"):
            deepface_functions.clear_cache()
    except Exception:
        pass


# ahh maybe needs a better name lol
class FacialRecognitionModel:
    MODEL_NAME = "Facenet"
    DETECTOR = "opencv"
    DISTANCE_METRIC = "cosine"

    prev_recognitions = []
    detection_thread: Thread | None = None
    should_exit = False # spaghetti code

    def __init__(
        self,
        db_path: Path | str = Path("data", "classification_data", "train_data"),
        liveness_model_path: Path | str = Path("src", "models", "liveness_model.keras"),
        liveness_threshold: float = 0.60,
    ) -> None:
        self.db_path = db_path
        self.liveness = LivenessDetector(liveness_model_path, threshold=liveness_threshold)
        self.gender_detector = GenderDetector("models/gender_model.h5")

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
            live_probability = 1.0
            is_live = True

            # Liveness / anti-spoofing is checked before trusting the identity match.
            # A phone-screen or printed-photo face should be labelled as SPOOF and blocked.
            
            gender_label = "Unknown"
            gender_confidence = 0.0

            if w > 0 and h > 0:
                face_crop = frame[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]

                try:
                    is_live, live_probability = self.liveness.check(face_crop)
                except Exception as e:
                    print(f"Liveness check failed: {e}")
                    is_live = False
                    live_probability = 0.0

                if is_live:
                    try:
                        gender_label, gender_confidence = self.gender_detector.predict(face_crop)
                    except Exception as e:
                        print(f"Gender prediction failed: {e}")
                        gender_label = "Unknown"
                        gender_confidence = 0.0

            if not is_live:
                name = "SPOOF / FAKE FACE"
            elif len(df) == 0:
                name = UNKNOWN_NAME
            else:
                identity_path = df.iloc[0]['identity'] # face_data may be unbound so just grab it again
                """
                pretty sick that you can get the closest image to you. (im adding that as a cool feature)
                we can display next to the webcam the image it is the most confident you resemble.
                for example: Oliver would look a LOT like Chris Hemsworth so put an image of shirtless Chris Hemsworth on screen.
                """
                name = os.path.basename(os.path.dirname(identity_path))

            results.append({
                'name': name,
                'box': (x, y, w, h),
                'is_live': is_live,
                'live_probability': live_probability,
                'gender': gender_label,
                'gender_confidence': gender_confidence,
            })
        return results
    
    @staticmethod
    def draw_boxes(frame, recognitions):
        for r in recognitions:
            x, y, w, h = r['box']
            name = r['name']
            is_live = r.get('is_live', True)
            live_probability = r.get('live_probability', 1.0)
            gender = r.get('gender', "Unknown")
            gender_confidence = r.get('gender_confidence', 0.0)
            if not is_live:
                color = (0, 0, 255)
                label = f"{name} | live={live_probability:.2f}"
            else:
                color = (0, 255, 0) if name != UNKNOWN_NAME else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            label = f"{name} | {gender}={gender_confidence:.2f} | live={live_probability:.2f}"
            cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_COMPLEX, 0.65, color, 2)
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
        
        
        while rval:
            rval, frame = vc.read()
            if standalone:
                key = cv2.waitKey(20)
                if key == 27: # exit on ESC
                    break
                if key == 100: # toggle detection on D
                    self.detection_active = not self.detection_active
                if key == ord('r'): # "i hate consistency" - Movi
                    self.register_face(frame)
            if self.detection_active:
                display_frame = self.async_detect(frame)
            else:
                display_frame = frame
            if self.should_exit:
                break
            if standalone:
                cv2.imshow(WINDOW_TITLE, display_frame)
            else:
                yield display_frame
        vc.release()
        if standalone:
            cv2.destroyAllWindows()

    def run_standalone(self, camera_index = 0):
        stream = self.run_stream(camera_index, True)
        next(stream)
    
    def stop_stream(self):
        self.should_exit = True

    def register_face(self, frame):
        name = input("Enter name for this new person: ").strip()

        if name == "":
            print("Registration cancelled. Name cannot be empty.")
            return

        person_folder = os.path.join(self.db_path, name)
        os.makedirs(person_folder, exist_ok=True)

        print("\nRegistration started.")
        print("Slowly turn your head left and right.")
        print("Try front view, side view, smiling, and different lighting.")
        print("Capturing 12 face images...\n")

        captured = 0

        while captured < 12:
            ret, frame = cv2.VideoCapture(0).read()

            if not ret:
                continue

            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                f"Registering {name}: {captured + 1}/12",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                "Slowly turn your head left and right",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Facial Recognition Attendance System", display_frame)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path = os.path.join(person_folder, f"{name}_{timestamp}.jpg")

            cv2.imwrite(image_path, frame)
            captured += 1

            cv2.waitKey(500)

        clear_deepface_cache()
        print(f"Registered {captured} images for {name}.")
if __name__ == "__main__":
    model = FacialRecognitionModel(Path("debug_data"))
    model.run_standalone()