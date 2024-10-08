import numpy as np
from skimage.feature import hog
from skimage import color
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from utils import load_data

def extract_hog_features(images):
    """
    Extract HOG features from a list of images.
    
    Args:
    - images: list of images as numpy arrays.
    
    Returns:
    - hog_features: array of HOG feature vectors.
    """
    hog_features = []
    
    for img in images:   
        features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                       block_norm='L2-Hys', visualize=False)
        hog_features.append(features)
    
    return np.array(hog_features)

# Load the images
X_images, y_labels, class_names = load_data('dataset_classification') # Same as with CNN

# Extract HOG features
hog_features = extract_hog_features(X_images)

# Convert labels to class indices
y_labels = np.argmax(y_labels, axis=1)

# Split the data into training (60%), validation (20%), and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(hog_features, y_labels, test_size=0.3, random_state=42)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate on the training set
y_train_pred = rf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training accuracy: {train_accuracy*100:.2f}%")

# Evaluate on the test set
y_test_pred = rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Generate the classification report for the test set
report = classification_report(y_test, y_test_pred, target_names=class_names)
print("Classification report on test set:")
print(report)
