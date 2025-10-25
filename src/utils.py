import numpy as np
from sklearn.metrics import confusion_matrix

def print_confusion_matrix(model, X_test, y_test):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print confusion matrix as text
    print("\nConfusion Matrix:")
    print("True Label \\ Predicted Label")
    header = " " * 10 + " ".join(f"{name[:8]:>8}" for name in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i][:8]:>10} " + " ".join(f"{val:>8}" for val in row)
        print(row_str)
