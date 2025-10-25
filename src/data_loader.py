import tensorflow as tf
import pickle
import os

def load_and_preprocess_data():
    # Load Fashion MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Optionally save to pickle for faster loading
    data = ((X_train, y_train), (X_test, y_test))
    os.makedirs('data', exist_ok=True)
    with open('data/fashion_mnist_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Image shape: {X_train[0].shape}")
    
    return (X_train, y_train), (X_test, y_test)