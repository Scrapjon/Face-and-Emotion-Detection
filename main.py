from face_emotion.face_recognition import FacialRecognitionModel

if __name__ == "__main__":
    model = FacialRecognitionModel()
    model.run_stream()