import tensorflow as tf
import numpy as np

# Load the saved model
loaded_modelk11 = tf.keras.models.load_model('saved_model/my_model_k11.keras')
loaded_modelk33 = tf.keras.models.load_model('saved_model/my_model_k33.keras')

# Prepare new input data
new_inputs = np.array([
    [0.45, 0.8, 0.1, 1, 0, 0],  # example input row
    # ... rows can be added if needed
])

# Make predictions
predictionsk11 = loaded_modelk11.predict(new_inputs)
print("Predictions:", predictionsk11)

predictionsk33 = loaded_modelk33.predict(new_inputs)
print("Predictions:", predictionsk33)
