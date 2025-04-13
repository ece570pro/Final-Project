import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Automatically get absolute path to saved models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_k11_path = os.path.join(BASE_DIR, "saved_model", "my_model_k11.keras")
model_k33_path = os.path.join(BASE_DIR, "saved_model", "my_model_k33.keras")

# Load models
model_k11 = tf.keras.models.load_model(model_k11_path)
model_k33 = tf.keras.models.load_model(model_k33_path)

# Custom scaling function
def custom_scale(X):
    X_scaled = X.copy()
    X_scaled[0] = 0.2 + X[0] * (0.8 - 0.2)       # vf
    X_scaled[1] = 0.2 + X[1] * (0.95 - 0.2)      # width (Wy/Sy)
    X_scaled[2] = 0.008 + X[2] * (0.5 - 0.008)   # thickness (Ty/Sy)
    return X_scaled

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract input features
        vf = float(data.get('vf', data['fiber_volume_fraction']))
        width = float(data.get('yarn_width', data['width']))
        thickness = float(data.get('yarn_thickness', data['thickness']))
        thickness = float(data['thickness'])
        p = float(data.get('p', data.get('spacing', 1.0)))
        t = float(data.get('t', data.get('textile_thickness', 1.0)))
        h = float(data.get('h', data.get('height', 1.0)))

        # Create input array and scale
        X_input = np.array([vf, width, thickness, p, t, h])
        X_scaled = custom_scale(X_input)
        X_scaled = X_scaled.reshape(1, -1)

        # Predict k11 and k33
        k11_pred = float(model_k11.predict(X_scaled)[0][0])
        k33_pred = float(model_k33.predict(X_scaled)[0][0])

        return jsonify({
            "k11": round(k11_pred, 6),
            "k33": round(k33_pred, 6)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
