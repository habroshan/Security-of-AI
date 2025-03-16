# test_api.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
import numpy as np
import requests
from tensorflow.keras.datasets import mnist
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

def test_target_model():
    # Load a test image
    print("Loading test data...")
    (_, _), (x_test, y_test) = mnist.load_data()
    test_image = x_test[0].astype('float32') / 255.0  # Normalize
    true_label = y_test[0]
    
    print(f"\nTesting with image (true label: {true_label})")
    
    try:
        # Send request to API
        response = requests.post(
            'http://localhost:5000/predict',
            json={'image': test_image.tolist()}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Print formatted results
            print("\nPrediction Results:")
            print(f"Model Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            
            print("\nTop 3 predictions:")
            for pred in result['top_3_predictions']:
                print(f"Digit {pred['digit']}: {pred['confidence']:.4f}")
                
            # Additional analysis
            print("\nPrediction Analysis:")
            if result['prediction'] == true_label:
                print("✓ Correct prediction!")
            else:
                print(f"✗ Incorrect prediction (true label: {true_label})")
            
            if result['confidence'] > 0.9:
                print("! High confidence prediction")
            
            # Test multiple predictions
            print("\nTesting multiple predictions...")
            test_batch = x_test[:5].astype('float32') / 255.0
            predictions = []
            for img in test_batch:
                resp = requests.post(
                    'http://localhost:5000/predict',
                    json={'image': img.tolist()}
                )
                if resp.status_code == 200:
                    predictions.append(resp.json()['prediction'])
            
            print(f"Predictions for 5 images: {predictions}")
            print(f"True labels: {y_test[:5]}")
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the Flask server (app.py) is running.")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")

    # Test health endpoint
    try:
        health_response = requests.get('http://localhost:5000/health')
        print(f"\nAPI Health Status: {health_response.json()['status']}")
    except:
        print("\nCould not check API health status")

if __name__ == "__main__":
    print("Starting API test...")
    test_target_model()
