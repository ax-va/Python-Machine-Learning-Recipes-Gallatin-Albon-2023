"""
Save a trained TensorFlow model and load it elsewhere.
->
Save the model using the TensorFlow save method in the `keras` format
or in the SavedModel (protobuf) format.

See also:
- TensorFlow: Save, serialize, and export models
https://www.tensorflow.org/guide/keras/serialization_and_saving

- TensorFlow: The SavedModel format on disk
https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf_version = tf.__version__.replace(".", "_")

# Set random seed
np.random.seed(0)

# Create model with one hidden layer
input_layer = keras.Input(shape=(10,))
hidden_layer = keras.layers.Dense(10)(input_layer)
output_layer = keras.layers.Dense(1)(hidden_layer)
model = keras.Model(input_layer, output_layer)
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
x_train = np.random.random((1000, 10))
y_train = np.random.random((1000, 1))
model.fit(x_train, y_train)

# Save the model as `model_tf.keras` in the `models` directory
model.save(f"models/model_tf_{tf_version}.keras")

# Save the model in the SavedModel format in the `models/saved_model` directory
model.export('models/saved_model')

# Load neural network
model = keras.models.load_model(f"models/model_tf_{tf_version}.keras")
