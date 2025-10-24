# Fashion MNIST Classifier

A simple project to classify images from the Fashion MNIST dataset using a neural network.

## Project Structure
- `data/`: Stores the dataset in `.pkl` format.
- `models/`: Stores the trained model in `.pkl` format.
- `src/`: Contains Python modules for data loading, model definition, training, and utilities.
- `main.py`: Main script to run the project.
- `requirements.txt`: Lists required dependencies.

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   python main.py
   ```

## Requirements
- Python 3.8+
- See `requirements.txt` for package dependencies.

## Output
- The script trains a neural network on the Fashion MNIST dataset.
- Outputs training progress, test accuracy, and a confusion matrix plot.
- Saves the trained model to `models/fashion_mnist_model.pkl`.
