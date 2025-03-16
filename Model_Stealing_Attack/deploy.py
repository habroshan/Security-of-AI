# Modified app.py with better validation
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model with validation
try:
    model = tf.keras.models.load_model('models/target_mnist_model.keras')
    
    # Validate model
    sample_input = np.random.rand(1, 28, 28, 1)  # Create sample input
    sample_pred = model.predict(sample_input, verbose=0)
    
    if len(np.unique(np.argmax(sample_pred, axis=1))) == 1:
        logger.warning("Model might be biased - predicting same class for random input")
    
    logger.info(f"Model architecture: {model.summary()}")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def preprocess_image(image_data):
    """Preprocess image data for model input with validation"""
    # Convert to float and normalize
    image = np.array(image_data, dtype='float32')
    
    # Check if image needs normalization
    if image.max() > 1.0:
        image = image / 255.0
    
    # Validate image values
    if image.min() < 0 or image.max() > 1:
        logger.warning(f"Image values out of range: min={image.min()}, max={image.max()}")
    
    # Ensure correct shape
    if len(image.shape) == 2:
        image = image.reshape(1, 28, 28, 1)
    elif len(image.shape) == 3:
        image = image.reshape(-1, 28, 28, 1)
    
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        image_data = np.array(data['image'])
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Validate prediction
        pred_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        logger.info(f"Prediction: {pred_class}, Confidence: {confidence}")
        
        # Get top 3 predictions
        top_3_idx = np.argsort(prediction[0])[-3:][::-1]
        top_3_values = prediction[0][top_3_idx]
        
        # Prepare response
        response = {
            'prediction': int(pred_class),
            'confidence': confidence,
            'top_3_predictions': [
                {'digit': int(idx), 'confidence': float(conf)}
                for idx, conf in zip(top_3_idx, top_3_values)
            ],
            'full_probabilities': prediction[0].tolist(),
            'input_shape': processed_image.shape,
            'input_range': {
                'min': float(processed_image.min()),
                'max': float(processed_image.max())
            }
        }
        
        logger.info(f"Prediction made successfully: {response['prediction']}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Test prediction on a sample image
        sample_input = np.random.rand(1, 28, 28, 1)
        test_pred = model.predict(sample_input, verbose=0)
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'test_prediction_shape': test_pred.shape[1]
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
