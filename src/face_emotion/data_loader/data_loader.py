from pathlib import Path
import tensorflow as tf

BATCH_SIZE = 32 # change later if needed
IMAGE_SIZE = 64 # width and height
SEED = 42       # keep consistent

DATA_ROOT_PATH = Path("data")
CLF_PATH = Path(DATA_ROOT_PATH, "classification_data")
CLF_TEST = Path(CLF_PATH, "test_data")
CLF_TRAIN = Path(CLF_PATH, "train_data")
CLF_VAL = Path(CLF_PATH, "val_data")
VER_PATH = Path(DATA_ROOT_PATH, "verification_data")


def load_dataset(path: Path | str) -> tf.data.Dataset:
    dataset: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory( # idk why pylance doesn't see this function.
        path,
        seed = SEED,
        image_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE
    )
    return dataset