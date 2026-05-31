import os
import pickle
import warnings
from pathlib import Path
from threading import Thread
from datetime import datetime
from typing import Optional
import cv2
import numpy as np
from face_emotion.anti_spoofing import LivenessDetector
from face_emotion.emotion_recognition.emotion_detector import EmotionDetector
from face_emotion.face_recognition.face_model import FaceRecognitionClient
from face_emotion.gender_detection.gender_model import GenderDetector
from face_emotion.rock_paper_scissors.rock_paper_scissors import Rock_Paper_Scissors

CAMERA_FPS_CAP = 30

UNKNOWN_NAME = "WHO ARE YOU???"
WINDOW_TITLE = "IMAGE RECOGNITION WOAHHHHH!"

# cosine similarity threshold. same person if score >= this. tune if recognition is too strict/loose
COSINE_THRESHOLD = 0.5

# pkl file suffix appended to db_path to store the embedding cache
DB_CACHE_SUFFIX = "_embeddings.pkl"

# image extensions we bother scanning when building the DB
SUPPORTED_IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# opencv's built-in haar cascade for frontal face detection
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"



def _cosine_sim(a, b) -> float:
    # dot product of two L2-normalised vectors == cosine similarity. face_client already normalises.
    return float(np.dot(np.asarray(a, dtype='float32'), np.asarray(b, dtype='float32')))


# ahh maybe needs a better name lol
class FacialRecognitionModel:
    prev_recognitions = []
    detection_thread: Thread | None = None
    should_exit = False  # spaghetti code

    def __init__(
        self,
        db_path: Path | str = Path("data", "classification_data", "train_data"),
        liveness_model_path: Path | str = Path("src", "models", "liveness_model.keras"),
        emotion_model_path: Path | str = Path(
            "src", "face_emotion", "emotion_recognition", "fine_tuned_models", "ft_emotion_model.h5"
        ),
        liveness_threshold: float = 0.60,
        cosine_threshold: float = COSINE_THRESHOLD,
        do_rpc: Optional[bool] = False
    ) -> None:
        self.db_path = Path(db_path)
        self.cosine_threshold = cosine_threshold
        self.detection_active = True

        # liveness / anti-spoofing
        self.liveness = LivenessDetector(liveness_model_path, threshold=liveness_threshold)
        
        # emotion detector (separate model, 48x48 grayscale -> 7-class softmax)
        self.emotion_detector = EmotionDetector(emotion_model_path)

        # rock paper scissors (seperat model, 640x640, 3-class)
        self.do_rpc = do_rpc
        self.rock_paper_scissors = Rock_Paper_Scissors()
        
        # our custom fine-tuned VGGFace model for face recognition
        print("[FacialRecognitionModel] Loading face recognition client...")
        self.face_client = FaceRecognitionClient()

        # haar cascade replaces deepface's detector. no extra dependencies needed, cv2 ships with it
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from {HAAR_CASCADE_PATH}. "
                "Something is very wrong with your opencv install."
            )

        # embedding DB: {person_name: [emb1, emb2, ...]}
        self.embedding_db: dict[str, list] = {}
        self._pkl_path = Path(str(self.db_path) + DB_CACHE_SUFFIX)
        self._load_or_build_db()

    # -- preprocessing --

    def _preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """resize to 224x224, BGR->RGB, add batch dim, cast to float32. ready for face_client.forward()"""
        resized = cv2.resize(face_bgr, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb, axis=0).astype('float32')

    # -- embedding DB management --

    def _load_or_build_db(self):
        """load cached pkl if it exists, otherwise scan db_path and build from scratch"""
        if self._pkl_path.exists():
            try:
                with open(self._pkl_path, 'rb') as f:
                    self.embedding_db = pickle.load(f)
                print(
                    f"[FacialRecognitionModel] Loaded {len(self.embedding_db)} people "
                    f"from cache at {self._pkl_path}"
                )
                return
            except Exception as e:
                print(f"[FacialRecognitionModel] Cache load failed ({e}), rebuilding...")

        print(
            f"[FacialRecognitionModel] Building embedding DB from {self.db_path}... "
            "(might take a bit for large datasets)"
        )
        self.embedding_db = self._build_embedding_db()
        self._save_db()
        print(f"[FacialRecognitionModel] DB ready with {len(self.embedding_db)} people")

    def _build_embedding_db(self) -> dict[str, list]:
        """scan db_path for person_name/image.jpg structure and compute embeddings for everything"""
        db: dict[str, list] = {}

        if not self.db_path.exists():
            warnings.warn(
                f"[FacialRecognitionModel] db_path {self.db_path} does not exist. "
                "Starting with an empty DB."
            )
            return db

        for person_dir in sorted(self.db_path.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            embeddings = []

            for img_file in sorted(person_dir.iterdir()):
                if img_file.suffix.lower() not in SUPPORTED_IMG_EXTENSIONS:
                    continue
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                try:
                    tensor = self._preprocess_face(img)
                    emb = self.face_client.forward(tensor)
                    embeddings.append(emb)
                except Exception as e:
                    print(f"  [skip] {img_file.name}: {e}")

            if embeddings:
                db[name] = embeddings
                print(f"  {name}: {len(embeddings)} embeddings")

        return db

    def _save_db(self):
        """pickle the embedding dict to disk so we dont recompute every startup"""
        try:
            with open(self._pkl_path, 'wb') as f:
                pickle.dump(self.embedding_db, f)
        except Exception as e:
            print(f"[FacialRecognitionModel] Failed to save DB cache: {e}")

    def _identify(self, embedding) -> tuple[str, float]:
        """
        compare embedding against all stored embeddings.
        returns (best_name, best_score). returns UNKNOWN_NAME if nothing beats cosine_threshold.
        """
        best_name = UNKNOWN_NAME
        best_score = -1.0

        for name, stored_embeddings in self.embedding_db.items():
            for stored_emb in stored_embeddings:
                score = _cosine_sim(embedding, stored_emb)
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < self.cosine_threshold:
            return UNKNOWN_NAME, best_score
        return best_name, best_score

    # -- core detection --

    def detect(self, frame) -> list[dict]:
        """
        detects faces (duh!)
        runs haar cascade -> liveness -> emotion -> identity on each detected face.
        returns list of dicts: {name, box, is_live, live_probability, emotion, emotion_probs}
        """
        results = []

        if frame is None or frame.size == 0:
            return results

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            face_crop = frame[max(0, y):y + h, max(0, x):x + w]

            if face_crop.size == 0:
                continue

            # liveness / anti-spoofing. phone screens and printed photos go here
            is_live = True
            live_probability = 1.0
            try:
                is_live, live_probability = self.liveness.check(face_crop)
            except Exception as e:
                print(f"Liveness check failed: {e}")
                is_live = False
                live_probability = 0.0

            # emotion
            emotion_label = 'unknown'
            emotion_probs: dict = {}
            if self.emotion_detector.is_available():
                try:
                    emotion_label, emotion_probs = self.emotion_detector.detect(face_crop)
                except Exception as e:
                    print(f"Emotion detection failed: {e}")

            # identity recognition
            if not is_live:
                name = "SPOOF / FAKE FACE"
            elif len(self.embedding_db) == 0:
                name = UNKNOWN_NAME
            else:
                try:
                    tensor = self._preprocess_face(face_crop)
                    embedding = self.face_client.forward(tensor)
                    name, _ = self._identify(embedding)
                except Exception as e:
                    print(f"Recognition failed: {e}")
                    name = UNKNOWN_NAME

            # rock paper scissors
            rpc_gesture = ""
            if self.do_rpc:
                try:
                    prediciton = self.rock_paper_scissors.predict(frame)
                    match prediciton:
                        case 0: rpc_gesture = "paper"
                        case 1: rpc_gesture = "rock"
                        case 2: rpc_gesture = "scissors"
                except Exception as e:
                    print(f"Rock Paper Scissors failed: {e}")
                    rpc_gesture = "none"
            
            results.append({
                'name': name,
                'box': (int(x), int(y), int(w), int(h)),
                'is_live': is_live,
                'live_probability': live_probability,
                'emotion': emotion_label,
                'emotion_probs': emotion_probs,
                'rpc_gesture': rpc_gesture,
            })

        return results

    @staticmethod
    def draw_boxes(frame, recognitions):
        for r in recognitions:
            x, y, w, h = r['box']
            name = r['name']
            is_live = r.get('is_live', True)
            live_probability = r.get('live_probability', 1.0)
            emotion = r.get('emotion', 'unknown')

            if not is_live:
                color = (0, 0, 255)  # red for spoof
            else:
                color = (0, 255, 0) if name != UNKNOWN_NAME else (0, 165, 255)  # green or orange

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            
            gender, gender_confidence = "GENDERLESS", 1 # placeholder
            label = f"{name} | {gender}={gender_confidence:.2f} | live={live_probability:.2f}"
            cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_COMPLEX, 0.65, color, 2)
            # emotion label goes just below the name/liveness text
            cv2.putText(frame, f"feeling: {emotion}", (x, label_y + 22), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        return frame

    def detect_and_assign(self, frame):
        self.prev_recognitions = self.detect(frame)

    def async_detect(self, frame):
        # Filter out NoneType
        if self.detection_thread is None:
            self.detection_thread = Thread(None, self.detect_and_assign, args=(frame,))
            self.detection_thread.start()

        # Check process isnt overlapping
        elif not self.detection_thread.is_alive():
            self.detection_thread = Thread(None, self.detect_and_assign, args=(frame,))
            self.detection_thread.start()

        return self.draw_boxes(frame, self.prev_recognitions)

    # -- registration --

    def register_face(self, frame, name: str):
        """
        register a single frame under the given name.
        call this once per captured frame - the UI handles looping and delays.
        detects a face, crops it, computes the embedding, adds to the in-memory DB, saves pkl.
        """
        if not name or not name.strip():
            print("register_face: skipping, name is empty")
            return

        person_folder = self.db_path / name
        person_folder.mkdir(parents=True, exist_ok=True)

        # always save the full frame as an image regardless of whether we find a face
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = person_folder / f"{name}_{timestamp}.jpg"
        cv2.imwrite(str(image_path), frame)

        # detect face to get a clean crop for the embedding
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            print(f"[register_face] No face detected for {name}, image saved but no embedding added")
            return

        # biggest face = most likely the person standing in front of the camera
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = frame[max(0, y):y + h, max(0, x):x + w]

        if face_crop.size == 0:
            return

        try:
            tensor = self._preprocess_face(face_crop)
            emb = self.face_client.forward(tensor)

            if name not in self.embedding_db:
                self.embedding_db[name] = []
            self.embedding_db[name].append(emb)
            self._save_db()
            print(f"[register_face] {name}: now has {len(self.embedding_db[name])} embeddings")
        except Exception as e:
            print(f"[register_face] Embedding failed for {name}: {e}")

        clear_deepface_cache()

    # -- stream helpers --

    def run_stream(self, camera_index=0, standalone=False):
        """
        Runs the video stream which is directly connected to the model.
        Args:
            camera_index (int): The index of the camera on the machine that the program will capture from.
            standalone (bool): True = runs as its own cv2 window. False = yields frames for a UI to consume.
        """
        self.prev_recognitions = []
        self.detection_active = True
        if standalone:
            cv2.namedWindow(WINDOW_TITLE)
        vc = cv2.VideoCapture(camera_index)

        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
            raise RuntimeError("NO CAMERA AAAAAAAAH (maybe try a different camera_index value...)")

        while rval:
            rval, frame = vc.read()
            if standalone:
                key = cv2.waitKey(20)
                if key == 27:  # ESC to exit
                    break
                if key == 100:  # D to toggle detection
                    self.detection_active = not self.detection_active
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

    def run_standalone(self, camera_index=0):
        stream = self.run_stream(camera_index, True)
        next(stream)

    def stop_stream(self):
        self.should_exit = True


if __name__ == "__main__":
    model = FacialRecognitionModel(Path("debug_data"))
    model.run_standalone()