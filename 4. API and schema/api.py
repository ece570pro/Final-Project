from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your ANN models
model_k11 = tf.keras.models.load_model('saved_model/my_model_k11.keras')
model_k33 = tf.keras.models.load_model('saved_model/my_model_k33.keras')

def preprocess_input(params):
    # Extract raw parameters from the input JSON
    vf = params['vf']  # Volume fraction in [0, 1]
    yarn_width = params['yarn_width']
    yarn_thickness = params['yarn_thickness']
    yarn_spacing = params['yarn_spacing']
    
    # Get the weave pattern as a string and convert to lower case for consistency.
    weave_pattern = params.get('weave_pattern', 'plain').lower()

    # Compute normalized ratios for yarn width and thickness relative to spacing.
    width_ratio = yarn_width / yarn_spacing
    thickness_ratio = yarn_thickness / yarn_spacing

    # Scale the values using linear scaling to the ranges used during training:
    # Volume fraction: [0,1] -> [0.2, 0.8]
    scaled_vf = 0.2 + vf * (0.8 - 0.2)
    # Yarn Width/Spacing: [0,1] -> [0.2, 0.95]
    scaled_width = 0.2 + width_ratio * (0.95 - 0.2)
    # Yarn Thickness/Spacing: [0,1] -> [0.008, 0.5]
    scaled_thickness = 0.008 + thickness_ratio * (0.5 - 0.008)

    # Map the weave pattern to p, t, h encoding:
    weave_mapping = {
        "plain": (1, 0, 0),
        "twill": (0, 1, 0),
        "5 harness satin": (0, 0, 1)
    }
    p, t, h = weave_mapping.get(weave_pattern, (1, 0, 0))  # Default to plain if not found

    # Build the input vector for the ANN: 
    # The expected order is [vf, width, thickness, p, t, h]
    input_vector = np.array([[scaled_vf, scaled_width, scaled_thickness, p, t, h]])
    return input_vector

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Thermal Conductivity Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON input from the request
    params = request.get_json()
    
    # Preprocess the input parameters
    input_data = preprocess_input(params)
    
    # Make predictions using both ANN models
    pred_k11 = model_k11.predict(input_data)[0, 0]
    pred_k33 = model_k33.predict(input_data)[0, 0]
    
    return jsonify({'k11': float(pred_k11), 'k33': float(pred_k33)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
