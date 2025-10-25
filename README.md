## Project Description

The **Fashion MNIST Classifier** is a machine learning project that classifies 28x28 grayscale images of clothing from the Fashion MNIST dataset into 10 categories using a TensorFlow neural network. The project is designed for console-only output, featuring modular code for data loading, model building, training, evaluation, and prediction. It prints a confusion matrix in text format to evaluate performance, saves the trained model, and allows users to make predictions on test images by entering an index.

## Project Structure

```
fashion_mnist_classifier/
├── data/
│   └── fashion_mnist_data.pkl  # Pickled dataset (optional, for caching)
├── models/
│   └── fashion_mnist_model.pkl  # Trained model
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── model.py           # Model definition
│   ├── train.py           # Training and evaluation logic
│   ├── utils.py           # Utility functions for console-based metrics
│   └── predict.py         # Prediction on user-provided data
├── main.py                # Main script to run the project
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Setup

1. **Create a virtual environment** (recommended with Anaconda):
   ```bash
   conda create -n fashion_mnist python=3.8
   conda activate fashion_mnist
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the project**:
   ```bash
   python main.py
   ```