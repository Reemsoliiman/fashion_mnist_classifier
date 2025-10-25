from tensorflow import keras

def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def save_model(model, model_path='models/fashion_mnist_model.h5'):
    """Save the Keras model properly using .h5 format."""
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path='models/fashion_mnist_model.h5'):
    """Load the saved Keras model from the specified path."""
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None