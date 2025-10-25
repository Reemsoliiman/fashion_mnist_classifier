# Fashion MNIST Classifier

A neural network classifier for the Fashion MNIST dataset using TensorFlow/Keras. This project trains a model to classify 28x28 grayscale images of clothing into 10 categories, with console-based output for predictions and evaluation metrics, including a text-based confusion matrix.

## Project Structure

```
fashion_mnist_classifier/
├── data/
│   └── fashion_mnist_data.pkl  # Pickled dataset (optional, for caching)
├── models/
│   └── fashion_mnist_model.h5  # Trained model
├── src/
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── model.py          # Model architecture and save/load functions
│   ├── train.py          # Training and evaluation functions
│   ├── predict.py        # Prediction functions for test images
│   └── utils.py          # Utility functions (e.g., confusion matrix)
├── main.py               # Main execution script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Reemsoliiman/fashion_mnist_classifier.git
   cd fashion_mnist_classifier
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Run the main script to train and test the model:**
```bash
python main.py
```

This will:
- Download the Fashion MNIST dataset (first run only)
- Preprocess and normalize the data
- Train a neural network for 10 epochs
- Save the trained model to `models/fashion_mnist_model.h5`
- Display test accuracy and a text-based confusion matrix
- Show predictions for test image indices 0, 1, and 2
- Prompt for interactive predictions on test images (enter an index from 0 to 9999 or 'q' to quit)

## Dataset

The Fashion MNIST dataset contains 70,000 grayscale images (28x28 pixels) in 10 categories:
- T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

The dataset (~30MB) is automatically downloaded by TensorFlow on the first run and optionally cached as a pickle file in the `data/` directory.

## Model Architecture

- **Input**: 28x28 grayscale images
- **Flatten layer**: Converts 2D image to 1D vector
- **Dense layer**: 100 neurons with ReLU activation
- **Output layer**: 10 neurons with Softmax activation
- **Optimizer**: Adam
- **Loss function**: Sparse categorical crossentropy
- **Metric**: Accuracy

## Notes

- The project uses console-only output for simplicity.
- Data (`data/fashion_mnist_data.pkl`) and model files (`models/fashion_mnist_model.h5`) are excluded from Git via `.gitignore`.
- The dataset is automatically downloaded by TensorFlow; no manual download is required.
- The model is saved in HDF5 format (`.h5`) for compatibility with Keras.
- TensorFlow oneDNN optimizations are disabled to suppress warnings (`TF_ENABLE_ONEDNN_OPTS=0`).

## Requirements

Key dependencies (listed in `requirements.txt`):
- TensorFlow
- NumPy
- Scikit-learn