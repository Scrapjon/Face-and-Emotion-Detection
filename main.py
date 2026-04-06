from io import StringIO
import dotenv
dotenv.load_dotenv(stream=StringIO("TF_ENABLE_ONEDNN_OPTS=0")) # bandaid solution for annoying errors, if anyone can fix this please do.
import tensorflow as tf

clf_test_data = tf.keras.utils.image_dataset_from_directory(
    "data/classification_data/test_data",
    seed=42,
    image_size = (150, 150),
    batch_size=32
)

print(clf_test_data)