import tensorflow as tf
import numpy as np

def process_image(image):
    """
    Takes an image (NumPy array) and returns a processed image (224, 224, 3) NumPy array.
    - Converts to TensorFlow Tensor
    - Resizes to 224x224
    - Normalizes to 0-1
    - Converts back to NumPy array
    """
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image.numpy()
