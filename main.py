import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warning
from src.data_loader import load_and_preprocess_data
from src.model import build_model, save_model, load_model
from src.train import train_model, evaluate_model
from src.utils import print_confusion_matrix
from src.predict import predict_image

def main():
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
    
    # Build model
    model = build_model()
    
    # Train model
    train_model(model, X_train, y_train, epochs=10, batch_size=32)
    
    # Evaluate model
    test_accuracy = evaluate_model(model, X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model properly using Keras format
    os.makedirs('models', exist_ok=True)
    save_model(model, 'models/fashion_mnist_model.h5')
    
    # Print confusion matrix
    print_confusion_matrix(model, X_test, y_test)
    
    # Test predictions on a few test images to verify model
    print("\nTesting model on sample test images:")
    for idx in [0, 1, 2]:
        predict_image(model, X_test, y_test, image_index=idx)
    
    # Interactive prediction on test images
    while True:
        print("\nEnter a test image index (0-9999) or 'q' to quit:")
        choice = input("Your choice: ").strip().lower()
        
        if choice == 'q':
            break
        else:
            try:
                image_index = int(choice)
                predict_image(model, X_test, y_test, image_index=image_index)
            except ValueError:
                print("Error: Please enter a valid integer or 'q' to quit.")

if __name__ == "__main__":
    main()