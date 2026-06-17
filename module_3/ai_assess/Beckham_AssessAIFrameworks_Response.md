# CSC-114: Assess AI Frameworks
## MNIST Handwritten Digit Classifier — Written Response

*William Beckham | Fayetteville Technical Community College | Summer 2026*

---

## 1. Dataset Attributes and Target Value

The dataset used is MNIST (Modified National Institute of Standards and Technology), a benchmark computer-vision dataset built into Keras. It contains 70,000 grayscale images of handwritten digits (60,000 training, 10,000 test). Each image has the following attributes:

- **pixel_values** — a 28×28 grid of unsigned 8-bit integers (0–255) representing grayscale intensity. After normalization these become float32 values in [0.0, 1.0]. The reshaped tensor has dimensions (samples, 28, 28, 1), where 1 is the single color channel.
- **label (target)** — an integer in the range 0–9 indicating which digit the image depicts. This is the target value the model is trained to predict.

There are no additional feature columns; the only "features" are the raw pixel values, and the only label is the digit class.

---

## 2. Regression or Classification?

This is a **multi-class classification** problem. The model must assign each input image to one of ten discrete categories (digits 0 through 9). This is confirmed by the choice of loss function (`sparse_categorical_crossentropy`) and the output layer architecture (10 neurons with a softmax activation), both of which are canonical choices for classification tasks. A regression model would instead output a continuous scalar and use mean squared error or a similar loss.

---

## 3. Optimizer Used and Rationale

The model was compiled with the **Adam optimizer** (Adaptive Moment Estimation). Adam combines two gradient-descent enhancements:

- **Momentum** — maintains a running average of past gradients to smooth the update direction and accelerate convergence in consistent gradient directions.
- **RMSProp-style adaptive learning rates** — scales the learning rate for each parameter individually based on the magnitude of recent gradients, preventing overshooting in dimensions with large gradients.

Adam is the default optimizer for most deep learning tasks because it converges quickly, is robust to hyperparameter choices, and requires minimal tuning. For image classification tasks like MNIST, Adam typically reaches near-optimal accuracy in fewer epochs than plain stochastic gradient descent (SGD), as was demonstrated in the Chapter 2 comparison where Adam/Keras achieved 98% versus 93% for a hand-built SGD implementation.

---

## 4. Training Epochs and Convergence

The model was trained for **5 epochs**. Training accuracy and loss across all epochs were as follows:

| Epoch | Training Accuracy | Training Loss | Notes |
|-------|------------------|---------------|-------|
| 1 | 92.16% | 0.2538 | Rapid initial learning |
| 2 | 97.72% | 0.0713 | Most of the gain concentrated here |
| 3 | 98.54% | 0.0479 | Diminishing returns begin |
| 4 | 98.82% | 0.0368 | |
| 5 | 99.12% | 0.0281 | Peak training accuracy |

The largest accuracy jump occurred between epoch 1 and epoch 2 (+5.56 percentage points). By epoch 5 the model reached 99.12% training accuracy with a loss of 0.0281, indicating strong convergence. Five epochs were sufficient; additional epochs would yield marginal returns and risk overfitting.

---

## 5. Best Accuracy / Lowest Loss Achieved

Evaluation on the held-out test set (10,000 images never seen during training) produced the following results:

- **Test accuracy: 98.9%**
- **Test loss: 0.0304**

The near-identical training and test accuracy (99.12% vs. 98.9%) indicates the model has generalized well and is not significantly overfitting. A 98.9% test accuracy means the model correctly classified 9,890 of the 10,000 test images, misclassifying only 110. This is a strong result for a 5-epoch run without any data augmentation or regularization techniques.

---

## 6. Save, Load, and Predict

Yes. The model was saved in Keras native format, reloaded from disk, and used to generate a prediction in the same Colab session:

```python
model.save("mnist_convnet.keras")

loaded_model = keras.saving.load_model("mnist_convnet.keras")

sample = test_images[0:1]  # shape (1, 28, 28, 1)
prediction = loaded_model.predict(sample)

print(f"Predicted digit: {np.argmax(prediction)}")
print(f"Actual label:    {test_labels[0]}")
```

**Output:**
```
Predicted digit: 7
Actual label:    7
```

The model correctly identified the first test image as a 7. The `.keras` format preserves the full model — architecture, weights, and optimizer state — so the loaded model behaves identically to the one trained in memory.

---

## 7. Additional Notes

### Architecture Choice

A Convolutional Neural Network (ConvNet) was selected over a dense network because spatial structure in images is meaningful — nearby pixels are related. Conv2D layers learn local filters (edges, curves, texture) that are translation-invariant, making them far more parameter-efficient than fully connected layers for image data.

### Layer-by-Layer Explanation

- **Conv2D (64 filters, 3×3) → MaxPooling2D:** Detects low-level features (edges, corners). MaxPooling halves the spatial dimensions, reducing computation and providing a degree of translation tolerance.
- **Conv2D (128 filters, 3×3) → MaxPooling2D:** Combines low-level features into mid-level patterns (curves, junctions characteristic of specific digit shapes).
- **Conv2D (256 filters, 3×3):** Detects high-level digit-specific shapes. No pooling here; the feature map is already small (3×3).
- **GlobalAveragePooling2D:** Averages each feature map to a single value, producing a 256-element vector. This replaces the traditional Flatten + Dense bottleneck, reduces parameters, and improves generalization.
- **Dense (10 neurons, softmax):** Converts the 256-element feature vector into a probability distribution over the 10 digit classes. The argmax of this output is the predicted digit.

### Total Parameters

The model has 372,234 trainable parameters (approximately 1.42 MB), distributed as:

- Conv2D 1: 640 parameters
- Conv2D 2: 73,856 parameters
- Conv2D 3: 295,168 parameters
- Dense output layer: 2,570 parameters

### CSC114Bot Consultation

Per the assignment requirement, CSC114Bot (the managed agent deployed to platform.claude.com for this course) was queried regarding appropriate datasets and model targets. The MNIST dataset was confirmed as an appropriate, pre-cleaned benchmark suitable for demonstrating classification with Keras, and Adam was recommended as the optimizer based on the Chapter 2 comparison results already documented in the course notebook work.
