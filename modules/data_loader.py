import tensorflow as tf

clf_test_data = tf.keras.utils.image_dataset_from_directory(
    "data/classification_data/test_data",
    seed=42,
    image_size = (150, 150),
    batch_size=32
)