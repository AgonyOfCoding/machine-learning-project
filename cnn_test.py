import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from utils import load_data

# Load data
X, y, class_names = load_data('dataset_classification')

# Normalize pixel values (0-1 range) and convert labels to categorical
X = X / 255.0
y = to_categorical(y, num_classes=len(class_names))

# Split the data into training (60%), validation (20%), and test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

def create_cnn_model(input_shape, num_classes):
    """
    Create a CNN model.
    
    Args:
    - input_shape: the shape of the input images (height, width, channels).
    - num_classes: the number of symbol classes.
    
    Returns:
    - model: the Sequental CNN model.
    """
    # 3 convolutional layers, 3 pooling layers, softmax final layer for classification
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Categorical cross-entropy as loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create CNN model
input_shape = X_train.shape[1:]
num_classes = len(class_names) # should be 10
model = create_cnn_model(input_shape, num_classes)

# Model summary
model.summary()

# ImageDataGenerator for real-time data augmentation
train_datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
train_datagen.fit(X_train)

# Train the model
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=64), # bigger batch size -> faster training if enough VRAM
                    validation_data=(X_val, y_val),
                    epochs=20,  # number of epochs
                    verbose=1)


# Plot the accuracy and loss curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.xticks(epochs)
plt.title('Training and Validation Accuracy')
plt.legend()

# loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.xticks(epochs)
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Class-specific results
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
print(report)