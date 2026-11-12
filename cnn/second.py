# --------------------------------------------------
# 1. Import libraries
# --------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------------------------------
# 2. Load and preprocess dataset
# --------------------------------------------------

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape to include channel dimension
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255.0

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# --------------------------------------------------
# 3. Build CNN model
# --------------------------------------------------

model = Sequential()
model.add(keras.Input(shape=(28, 28, 1)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# --------------------------------------------------
# 4. Compile and train model
# --------------------------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# --------------------------------------------------
# 5. Evaluate model
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# --------------------------------------------------
# 6. Confusion Matrix and Classification Report
# --------------------------------------------------

pred = model.predict(x_test)
y_pred = np.argmax(pred, axis=1)
y_actual = np.argmax(y_test, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_actual, y_pred))

print("Classification Report:")
print(classification_report(y_actual, y_pred))
