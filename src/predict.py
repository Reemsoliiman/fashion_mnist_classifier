import numpy as np

def predict_image(model, X_test, y_test, image_index):
    """Make a prediction on a test image."""
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Validate index
    if image_index < 0 or image_index >= len(X_test):
        print(f"Error: Invalid test image index {image_index}. Must be 0 to {len(X_test)-1}.")
        return
    
    # Get image (already normalized in data_loader)
    image = X_test[image_index]
    true_label = y_test[image_index]
    
    # Reshape for model input (1, 28, 28)
    image_input = image.reshape(1, 28, 28)
    
    # Make prediction
    try:
        probabilities = model.predict(image_input, verbose=0)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        predicted_label = class_names[predicted_class]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Print prediction results
    print(f"\nPrediction for test image index {image_index}:")
    print(f"Predicted class: {predicted_label}")
    print(f"True label: {class_names[true_label]}")
    print("\nConfidence scores for all classes:")
    for i, prob in enumerate(probabilities[0]):
        print(f"{class_names[i]:<15}: {prob:.4f}")