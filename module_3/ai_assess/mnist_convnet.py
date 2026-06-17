"""
CSC-114 – Assess AI Frameworks
MNIST Handwritten Digit Classifier — Convolutional Neural Network
Author: William Beckham
Dataset: MNIST (Modified National Institute of Standards and Technology)
Framework: Keras 3 with TensorFlow backend
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"   # Alternatives: "jax" or "torch"

import numpy as np
import keras
from keras import layers

# ---------------------------------------------------------------------------
# 1. Load and pre-process the MNIST dataset
# ---------------------------------------------------------------------------
# MNIST ships pre-split: 60,000 training images and 10,000 test images.
# Each image is a 28×28 grayscale pixel grid with a label 0–9.
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape to (samples, height, width, channels) — Conv2D expects a channel dim.
train_images = train_images.reshape((60000, 28, 28, 1))
test_images  = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values from uint8 [0, 255] to float32 [0.0, 1.0].
train_images = train_images.astype("float32") / 255
test_images  = test_images.astype("float32") / 255

# ---------------------------------------------------------------------------
# 2. Build the model — three Conv2D layers followed by a Dense classifier
# ---------------------------------------------------------------------------
inputs = keras.Input(shape=(28, 28, 1))

# Conv block 1: 64 filters, 3×3 kernel, ReLU activation → MaxPool halves spatial dims
x = layers.Conv2D(filters=64,  kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)

# Conv block 2: 128 filters
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)

# Conv block 3: 256 filters — GlobalAveragePooling replaces Flatten + Dense bottleneck
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)

# Output layer: 10 neurons (one per digit class), softmax converts to probabilities
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# ---------------------------------------------------------------------------
# 3. Compile — Adam optimizer, sparse categorical crossentropy loss
# ---------------------------------------------------------------------------
# sparse_categorical_crossentropy accepts integer labels directly (no one-hot needed).
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---------------------------------------------------------------------------
# 4. Train — 5 epochs, batch size 64
# ---------------------------------------------------------------------------
history = model.fit(train_images, train_labels, epochs=5, batch_size=64)

# ---------------------------------------------------------------------------
# 5. Evaluate on held-out test set
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.3f}")
print(f"Test loss:     {test_loss:.4f}")

# ---------------------------------------------------------------------------
# 6. Save the trained model in Keras native format
# ---------------------------------------------------------------------------
model.save("mnist_convnet.keras")
print("Model saved to mnist_convnet.keras")

# ---------------------------------------------------------------------------
# 7. Reload and run a single-sample prediction to verify save/load round-trip
# ---------------------------------------------------------------------------
loaded_model = keras.saving.load_model("mnist_convnet.keras")

sample     = test_images[0:1]            # shape (1, 28, 28, 1)
prediction = loaded_model.predict(sample)

print(f"Predicted digit: {np.argmax(prediction)}")
print(f"Actual label:    {test_labels[0]}")
