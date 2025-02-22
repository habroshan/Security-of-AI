"""
Model Deployment Script using Flask

This script deploys the trained MNIST digit classifier model as a REST API using Flask.
It provides endpoints for model information and image classification.

Usage:
    python deploy.py

Endpoints:
    GET /info - Returns information about the model
    POST /predict - Accepts an image and returns the prediction
    POST /predict_raw - Accepts raw pixel data and returns the prediction
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
MODEL = None
MODEL_INFO = None
CLASS_NAMES = None

def load_model():
    """Load the trained model and model information"""
    global MODEL, MODEL_INFO, CLASS_NAMES
    
    print("Loading model...")
    MODEL = tf.keras.models.load_model('saved_model/mnist_cnn_model.h5')
    
    print("Loading model information...")
    with open('saved_model/model_info.pkl', 'rb') as f:
        MODEL_INFO = pickle.load(f)
    
    CLASS_NAMES = MODEL_INFO['class_names']
    
    print("Model loaded successfully!")
    print(f"Model input shape: {MODEL_INFO['input_shape']}")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    print(f"Test accuracy: {MODEL_INFO['test_accuracy']}")

def preprocess_image(image_data):
    """
    Preprocess the input image for model prediction
    
    Args:
        image_data: Base64 encoded image data or file path
    
    Returns:
        Preprocessed image tensor
    """
    try:
        # Check if input is a base64 string
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Extract the base64 part
            image_data = image_data.split(',')[1]
            # Decode base64 to image
            image = Image.open(BytesIO(base64.b64decode(image_data)))
        elif isinstance(image_data, str):
            # If it's a file path
            image = Image.open(image_data)
        else:
            # If it's already a PIL Image
            image = image_data
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize image to the required input shape
        image = image.resize((28, 28))
        
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for the model (add batch and channel dimensions)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        
        return img_array
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

@app.route('/info', methods=['GET'])
def get_model_info():
    """Return information about the model"""
    response = {
        'model_name': 'MNIST Digit Classifier',
        'input_shape': MODEL_INFO['input_shape'],
        'classes': CLASS_NAMES,
        'test_accuracy': float(MODEL_INFO['test_accuracy']),
        'preprocessing': MODEL_INFO['preprocessing']
    }
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Process an image and return the predicted class
    
    Accepts:
        - JSON with 'image' field containing base64 encoded image
        - Form data with 'image' file
    
    Returns:
        JSON with prediction results
    """
    try:
        # Check request type
        if request.is_json:
            # Get base64 image from JSON
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400
            image_data = data['image']
        else:
            # Get image from form data
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            file = request.files['image']
            image_data = file
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = MODEL.predict(processed_image)[0]
        
        # Get top predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        
        # Get all class probabilities
        class_probabilities = {class_name: float(prob) 
                               for class_name, prob in zip(CLASS_NAMES, predictions)}
        
        # Return prediction
        response = {
            'predicted_class': CLASS_NAMES[predicted_class_idx],
            'confidence': confidence,
            'class_probabilities': class_probabilities
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_raw', methods=['POST'])
def predict_raw():
    """
    Endpoint for attacks to use directly with raw pixel data
    
    Accepts:
        JSON with 'pixels' field containing a list of pixel values (0-1)
    
    Returns:
        JSON with prediction results
    """
    try:
        data = request.get_json()
        
        if 'pixels' not in data:
            return jsonify({'error': 'No pixel data provided'}), 400
        
        # Convert to numpy array and reshape
        pixels = np.array(data['pixels'], dtype='float32')
        
        # Reshape to match model input (batch_size, height, width, channels)
        input_shape = tuple([1] + list(MODEL_INFO['input_shape']))
        
        if pixels.size != np.prod(MODEL_INFO['input_shape']):
            return jsonify({'error': f"Pixel data must have {np.prod(MODEL_INFO['input_shape'])} elements"}), 400
        
        pixels = pixels.reshape(input_shape)
        
        # Make prediction
        predictions = MODEL.predict(pixels)[0]
        
        # Get top predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        
        # Get all class probabilities
        class_probabilities = {class_name: float(prob) 
                               for class_name, prob in zip(CLASS_NAMES, predictions)}
        
        # Return prediction
        response = {
            'predicted_class': CLASS_NAMES[predicted_class_idx],
            'confidence': confidence,
            'class_probabilities': class_probabilities
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_gradient', methods=['POST'])
def get_gradient():
    """
    Endpoint to get the gradient of the loss with respect to the input
    This is useful for implementing gradient-based attacks like FGSM
    
    Accepts:
        JSON with 'pixels' field containing the input image and 'label' field with the true label
    
    Returns:
        JSON with gradient information
    """
    try:
        data = request.get_json()
        
        if 'pixels' not in data or 'label' not in data:
            return jsonify({'error': 'Both pixels and label must be provided'}), 400
        
        # Convert inputs to TensorFlow tensors
        pixels = np.array(data['pixels'], dtype='float32')
        input_shape = tuple([1] + list(MODEL_INFO['input_shape']))
        pixels = pixels.reshape(input_shape)
        
        # Convert label to one-hot encoding
        label = int(data['label'])
        label_one_hot = tf.one_hot(label, len(CLASS_NAMES))
        label_one_hot = tf.reshape(label_one_hot, (1, len(CLASS_NAMES)))
        
        # Get gradient using TensorFlow
        pixels_tensor = tf.convert_to_tensor(pixels)
        
        with tf.GradientTape() as tape:
            tape.watch(pixels_tensor)
            prediction = MODEL(pixels_tensor)
            loss = tf.keras.losses.categorical_crossentropy(label_one_hot, prediction)
        
        # Get the gradient
        gradient = tape.gradient(loss, pixels_tensor).numpy()
        
        # Return the gradient as a flattened list
        response = {
            'gradient': gradient.flatten().tolist(),
            'gradient_shape': gradient.shape
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Return a simple welcome message"""
    return jsonify({
        'message': 'MNIST Digit Classifier API is running',
        'endpoints': {
            'GET /info': 'Get model information',
            'POST /predict': 'Submit an image for classification',
            'POST /predict_raw': 'Submit raw pixel data for classification',
            'POST /get_gradient': 'Get gradient of loss with respect to input'
        }
    })

if __name__ == '__main__':
    # Load model before starting the server
    load_model()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
