# Fashion MNIST Classifier

A neural network classifier for the Fashion MNIST dataset using TensorFlow/Keras. This project trains a model to classify 28x28 grayscale images of clothing into 10 categories, with console-based output for predictions and evaluation metrics, including a text-based confusion matrix.

## Project Structure

```
fashion_mnist_classifier/
├── models/
│   └── fashion_mnist_model.h5  # Trained model (generated after training)
├── src/
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── model.py          # Model architecture and save/load functions
│   ├── train.py          # Training and evaluation functions
│   ├── predict.py        # Prediction functions for test images
│   └── utils.py          # Utility functions (e.g., confusion matrix)
├── main.py               # Main execution script
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Reemsoliiman/fashion_mnist_classifier.git
   cd fashion_mnist_classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Run the main script to train and test the model:**
```bash
python main.py
```

This will:
- Automatically download the Fashion MNIST dataset (first run only, downloaded by TensorFlow/Keras)
- Preprocess and normalize the data
- Train a neural network for 10 epochs
- Evaluate the model on the test set
- Save the trained model to `models/fashion_mnist_model.h5`
- Display test accuracy and a text-based confusion matrix
- Show predictions for test image indices 0, 1, and 2
- Enter interactive mode where you can test predictions on any test image (enter an index from 0 to 9999 or 'q' to quit)

## Dataset

The Fashion MNIST dataset contains 70,000 grayscale images (28x28 pixels) in 10 categories:
- **T-shirt/top** - **Trouser** - **Pullover** - **Dress** - **Coat**
- **Sandal** - **Shirt** - **Sneaker** - **Bag** - **Ankle boot**

**Training set:** 60,000 images  
**Test set:** 10,000 images

The dataset is automatically downloaded by TensorFlow/Keras on the first run and cached in your system's Keras data