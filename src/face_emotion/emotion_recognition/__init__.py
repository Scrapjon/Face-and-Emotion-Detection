# NOTE: we intentionally do NOT import from emotion_model here.
# emotion_model.py is a training script with module-level code that downloads weights,
# loads datasets, and trains a model. importing it at runtime would be a disaster.
# if you need to retrain, run emotion_model.py directly as a script.
 
from face_emotion.emotion_recognition.emotion_detector import EmotionDetector
 
__all__ = ["EmotionDetector"]