# enhanced_model_test.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def test_target_model():
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model('models/target_mnist_model.keras')
    
    # Load test data
    print("Loading test data...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Test predictions
    print("\nTesting predictions...")
    predictions = model.predict(x_test[:100])
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = y_test[:100]
    
    # Detailed analysis
    print("\nDetailed Analysis:")
    print("1. Prediction Sums:")
    pred_sums = np.sum(predictions, axis=1)
    print(f"Mean sum: {np.mean(pred_sums):.4f}")
    print(f"Min sum: {np.min(pred_sums):.4f}")
    print(f"Max sum: {np.max(pred_sums):.4f}")
    
    print("\n2. Confidence Analysis:")
    confidences = np.max(predictions, axis=1)
    print(f"Mean confidence: {np.mean(confidences):.4f}")
    print(f"Min confidence: {np.min(confidences):.4f}")
    print(f"Max confidence: {np.max(confidences):.4f}")
    
    print("\n3. Sample Predictions (first 5):")
    for i in range(5):
        print(f"\nSample {i}:")
        print(f"True label: {true_classes[i]}")
        print(f"Predicted: {pred_classes[i]}")
        print(f"Confidence: {confidences[i]:.4f}")
        print("Distribution:", predictions[i])
    
    # Visualize predictions
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Confidence distribution
    plt.subplot(1, 3, 1)
    plt.hist(confidences, bins=20)
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    
    # Plot 2: Prediction sums
    plt.subplot(1, 3, 2)
    plt.hist(pred_sums, bins=20)
    plt.title('Prediction Sums')
    plt.xlabel('Sum')
    plt.ylabel('Count')
    
    # Plot 3: Sample image with predictions
    plt.subplot(1, 3, 3)
    plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
    plt.title(f'Sample Image\nTrue: {true_classes[0]}, Pred: {pred_classes[0]}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Check model weights
    print("\nModel Weights Analysis:")
    for layer in model.layers:
        if layer.weights:
            weights = layer.weights[0].numpy()
            print(f"\nLayer: {layer.name}")
            print(f"Weight shape: {weights.shape}")
            print(f"Weight stats: mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")
            print(f"Any NaN: {np.any(np.isnan(weights))}")
            print(f"Any Inf: {np.any(np.isinf(weights))}")

if __name__ == "__main__":
    test_target_model()