import os
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_dir):
    """
    Load and preprocess image and label data from the given directory.
    
    Args:
    - data_dir (str): the folder containing image class folders.
    
    Returns:
    - X: array of images.
    - y: array of corresponding labels.
    - class_names: list of all symbol classes
    """
    X, y = [], []
    class_names = sorted(os.listdir(data_dir))
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                img = plt.imread(img_path)
                
                # Convert to uint8 format?
                if img.dtype != np.uint8:
                    img = (255 * img).astype(np.uint8)
                
                
                X.append(img)
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, class_names