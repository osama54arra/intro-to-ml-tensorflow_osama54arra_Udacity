import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

# Load the dataset
dataset_name = 'oxford_flowers102'
(train_ds, val_ds, test_ds), ds_info = tfds.load(
    dataset_name,
    split=['train', 'validation', 'test'],
    with_info=True,
    as_supervised=True
)

# Number of classes
num_classes = ds_info.features['label'].num_classes

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, label

# Apply preprocessing
batch_size = 32
train_ds = train_ds.map(preprocess).batch(batch_size).shuffle(1000)
val_ds = val_ds.map(preprocess).batch(batch_size)
test_ds = test_ds.map(preprocess).batch(batch_size)

# Load MobileNet feature extractor
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3), trainable=False)

# Build the model
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
epochs = 10  # Adjust as needed
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Save the model
model.save('my_model.h5')

print("Model trained and saved as my_model.h5")
