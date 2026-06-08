# Chapter 2: The Mathematical Building Blocks of Neural Networks

## This chapter covers

- A first example of a neural network
- Tensors and tensor operations
- How neural networks learn via backpropagation and gradient descent

Understanding deep learning requires familiarity with many simple mathematical concepts: tensors, tensor operations, differentiation, gradient descent, and so on. Our goal in this chapter will be to build up your intuition about these notions without getting overly technical. In particular, we'll steer away from mathematical notation, which can introduce unnecessary barriers for those without any mathematics background, and isn't necessary to explain things well. The most precise, unambiguous description of a mathematical operation is its executable code.

To provide sufficient context for introducing tensors and gradient descent, we'll begin the chapter with a practical example of a neural network. Then we'll go over every new concept that's been introduced, point by point. Keep in mind that these concepts will be essential for you to understand the practical examples that will come in the following chapters!

After reading this chapter, you'll have an intuitive understanding of the mathematical theory behind deep learning, and you'll be ready to start diving into modern deep learning frameworks, in chapter 3.

---

> **Running the code in this book**
> 
> This book is full of runnable Python code. Each chapter is paired with a Jupyter notebook that contains all of the code from the chapter. A Jupyter notebook is a live Python scratch pad of sorts, where you can interactively run code, graph data, view images, and a lot more. You will gain a lot more practical knowledge from this book if you run and experiment with the code as you read.
> 
> By far the easiest way to set up a deep learning environment to run these notebooks is Google Colaboratory (or Colab for short), a hosted environment for Jupyter notebooks that has become the industry standard for ML practitioners. With Colab, you can run the code for this book interactively in the browser, connecting to cloud runtimes with configurable hardware. By default, the notebooks in this book will run on Colab's free GPU runtime.
> 
> If you would like, you can also run these notebooks locally on your own machine. A GPU is recommended, especially as you get to the larger and more compute-intensive models later in this book.
> 
> Instructions for running locally and on Colab, along with the code, can be found at https://github.com/fchollet/deep-learning-with-python-notebooks.

---

## 2.1 A First Look at a Neural Network

Let's look at a concrete example of a neural network that uses the machine learning library Keras to learn to classify handwritten digits. We will use Keras extensively throughout this book. It's a simple, high-level library that will allow us to stay focused on the concepts we would like to cover.

Unless you already have experience with Keras or similar libraries, you won't understand everything about this first example right away. That's fine. In a few sections, we'll review each element in the example and explain it in detail. So don't worry if some steps seem arbitrary or look like magic to you! We've got to start somewhere.

The problem we're trying to solve here is to classify grayscale images of handwritten digits (28 × 28 pixels) into their 10 categories (0 through 9). We'll use the MNIST dataset, a classic in the machine learning community, which has been around almost as long as the field itself and has been intensively studied. It's a set of 60,000 training images, plus 10,000 test images, assembled by the National Institute of Standards and Technology (the NIST in MNIST) in the 1980s. You can think of "solving" MNIST as the "Hello World" of deep learning—it's what you do to verify that your algorithms are working as expected. As you become a machine learning practitioner, you'll see MNIST come up over and over again, in scientific papers, blog posts, and so on.

> **NOTE:** In machine learning, a category in a classification problem is called a **class**. Data points are called **samples**. The class associated with a specific sample is called a **label**.

The MNIST dataset comes preloaded in Keras, in the form of a set of four NumPy arrays.

**Listing 2.1 — Loading the MNIST dataset in Keras**

```python
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

`train_images` and `train_labels` form the training set, the data that the model will learn from. The model will then be tested on the test set, `test_images` and `test_labels`. The images are encoded as NumPy arrays, and the labels are an array of digits, ranging from 0 to 9. The images and labels have a one-to-one correspondence.

> **NOTE:** NumPy is a highly popular Python library for numerical computation. It is rarely used to implement modern machine learning algorithms, due to lacking GPU and autodifferentiation support, but NumPy arrays are often used as a numerical data exchange format.

Let's look at the training data:

```python
>>> train_images.shape
(60000, 28, 28)
>>> len(train_labels)
60000
>>> train_labels
array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
```

And here's the test data:

```python
>>> test_images.shape
(10000, 28, 28)
>>> len(test_labels)
10000
>>> test_labels
array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)
```

The workflow will be as follows. First, we'll feed the neural network the training data, `train_images` and `train_labels`. The network will then learn to associate images and labels. Finally, we'll ask the network to produce predictions for `test_images`, and we'll verify whether these predictions match the labels from `test_labels`.

**Listing 2.2 — The network architecture**

```python
import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
```

The core building block of neural networks is the **layer**. You can think of a layer as a filter for data: some data goes in, and it comes out in a more useful form. Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand. Most of deep learning consists of chaining together simple layers that will implement a form of progressive data distillation. A deep learning model is like a sieve for data processing, made of a succession of increasingly refined data filters—the layers.

Here, our model consists of a sequence of two `Dense` layers, which are densely connected (also called fully connected) neural layers. The second (and last) layer is a 10-way softmax classification layer, which means it will return an array of 10 probability scores (summing to 1). Each score will be the probability that the current digit image belongs to one of our 10 digit classes.

To make the model ready for training, we need to pick three more things, as part of the **compilation** step:

- A **loss function** — How the model will be able to measure its performance on the training data and thus how it will be able to steer itself in the right direction.
- An **optimizer** — The mechanism through which the model will update itself based on the training data it sees, to improve its performance.
- **Metrics to monitor** during training and testing — Here, we'll only care about accuracy (the fraction of the images that were correctly classified).

**Listing 2.3 — The compilation step**

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

Before training, we'll preprocess the data by reshaping it into the shape the model expects and scaling it so that all values are in the [0, 1] interval.

**Listing 2.4 — Preparing the image data**

```python
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```

We're now ready to train the model, which in Keras is done via a call to the model's `fit()` method.

**Listing 2.5 — "Fitting" the model**

```python
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```

Two quantities are displayed during training: the loss of the model over the training data and the accuracy of the model over the training data. We quickly reach an accuracy of 0.989 (98.9%) on the training data.

**Listing 2.6 — Using the model to make predictions**

```python
>>> test_digits = test_images[0:10]
>>> predictions = model.predict(test_digits)
>>> predictions[0]
array([1.0726176e-10, 1.6918376e-10, 6.1314843e-08, 8.4106023e-06,
       2.9967067e-11, 3.0331331e-09, 8.3651971e-14, 9.9999106e-01,
       2.6657624e-08, 3.8127661e-07], dtype=float32)
```

Each number of index `i` in that array corresponds to the probability that digit image `test_digits[0]` belongs to class `i`. This first test digit has the highest probability score at index 7:

```python
>>> predictions[0].argmax()
7
>>> predictions[0][7]
0.99999106
>>> test_labels[0]
7
```

**Listing 2.7 — Evaluating the model on new data**

```python
>>> test_loss, test_acc = model.evaluate(test_images, test_labels)
>>> print(f"test_acc: {test_acc}")
test_acc: 0.9785
```

The test set accuracy turns out to be 97.8%—that's slightly lower than training accuracy (98.9%). This gap between training accuracy and test accuracy is an example of **overfitting**: the fact that machine learning models tend to perform worse on new data than on their training data. Overfitting is a central topic in chapter 5.

---

## 2.2 Data Representations for Neural Networks

In the previous example, we started from data stored in multidimensional NumPy arrays, also called **tensors**. In general, all current machine learning systems use tensors as their basic data structure. So what's a tensor?

At its core, a tensor is a container for data—usually numerical data. You may already be familiar with matrices, which are rank-2 tensors: tensors are a generalization of matrices to an arbitrary number of dimensions (note that in the context of tensors, a dimension is often called an **axis**).

### 2.2.1 Scalars (Rank-0 Tensors)

A tensor that contains only one number is called a **scalar** (or scalar tensor, rank-0 tensor, or 0D tensor). In NumPy, a `float32` or `float64` number is a scalar tensor. A scalar tensor has 0 axes (`ndim == 0`).

```python
>>> import numpy as np
>>> x = np.array(12)
>>> x
array(12)
>>> x.ndim
0
```

### 2.2.2 Vectors (Rank-1 Tensors)

An array of numbers is called a **vector** (or rank-1 tensor or 1D tensor). A rank-1 tensor has exactly one axis.

```python
>>> x = np.array([12, 3, 6, 14, 7])
>>> x
array([12,  3,  6, 14,  7])
>>> x.ndim
1
```

> **NOTE:** Don't confuse a 5D *vector* with a 5D *tensor*! A 5D vector has only one axis and has five dimensions along its axis, whereas a 5D tensor has five axes.

### 2.2.3 Matrices (Rank-2 Tensors)

An array of vectors is a **matrix** (or rank-2 tensor or 2D tensor). A matrix has two axes (often referred to as rows and columns).

```python
>>> x = np.array([[5, 78, 2, 34, 0],
...               [6, 79, 3, 35, 1],
...               [7, 80, 4, 36, 2]])
>>> x.ndim
2
```

### 2.2.4 Rank-3 Tensors and Higher-Rank Tensors

If you pack matrices in a new array, you obtain a rank-3 tensor (or 3D tensor), which you can visually interpret as a cube of numbers.

```python
>>> x = np.array([[[5, 78, 2, 34, 0],
...                [6, 79, 3, 35, 1],
...                [7, 80, 4, 36, 2]],
...               [[5, 78, 2, 34, 0],
...                [6, 79, 3, 35, 1],
...                [7, 80, 4, 36, 2]],
...               [[5, 78, 2, 34, 0],
...                [6, 79, 3, 35, 1],
...                [7, 80, 4, 36, 2]]])
>>> x.ndim
3
```

In deep learning, you'll generally manipulate tensors with ranks 0 to 4, although you may go up to 5 if you process video data.

### 2.2.5 Key Attributes

A tensor is defined by three key attributes:

- **Number of axes (rank)** — For instance, a rank-3 tensor has three axes. Also called `ndim` in Python libraries.
- **Shape** — A tuple of integers that describes how many dimensions the tensor has along each axis. For instance, a matrix might have shape `(3, 5)`.
- **Data type (dtype)** — The type of the data contained in the tensor; for instance, `float16`, `float32`, `float64`, `uint8`, `bool`, and so on.

Looking at the MNIST data:

```python
>>> train_images.ndim
3
>>> train_images.shape
(60000, 28, 28)
>>> train_images.dtype
uint8
```

So what we have here is a rank-3 tensor of 8-bit integers: an array of 60,000 matrices of 28 × 28 integers. Each such matrix is a grayscale image, with coefficients between 0 and 255.

**Listing 2.8 — Displaying the fourth digit**

```python
import matplotlib.pyplot as plt

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
```

```python
>>> train_labels[4]
9
```

### 2.2.6 Manipulating Tensors in NumPy

Selecting specific elements in a tensor is called **tensor slicing**.

```python
>>> my_slice = train_images[10:100]
>>> my_slice.shape
(90, 28, 28)

# Equivalent notation:
>>> my_slice = train_images[10:100, :, :]
>>> my_slice = train_images[10:100, 0:28, 0:28]

# Bottom-right corner of all images:
my_slice = train_images[:, 14:, 14:]

# Center crop:
my_slice = train_images[:, 7:-7, 7:-7]
```

### 2.2.7 The Notion of Data Batches

In general, the first axis (axis 0) in all data tensors you'll come across in deep learning will be the **samples axis**. Deep learning models don't process an entire dataset at once; rather, they break the data into small **batches**.

```python
# Batch size of 128:
batch = train_images[:128]

# Next batch:
batch = train_images[128:256]

# nth batch:
n = 3
batch = train_images[128 * n : 128 * (n + 1)]
```

### 2.2.8 Real-World Examples of Data Tensors

The data you'll manipulate will almost always fall into one of the following categories:

- **Vector data** — Rank-2 tensors of shape `(samples, features)`
- **Timeseries data or sequence data** — Rank-3 tensors of shape `(samples, timesteps, features)`
- **Images** — Rank-4 tensors of shape `(samples, height, width, channels)`
- **Video** — Rank-5 tensors of shape `(samples, frames, height, width, channels)`

**Vector data examples:**
- An actuarial dataset of 100,000 people with age, gender, and income → shape `(100000, 3)`
- A dataset of 500 text documents with word counts over 20,000 words → shape `(500, 20000)`

**Timeseries data examples:**
- Stock prices: every minute stores current price, highest, and lowest → 250 days of data has shape `(250, 390, 3)`
- Tweets encoded as sequences of characters → 1 million tweets has shape `(1000000, 280, 128)`

**Image data:**
- A batch of 128 grayscale images of size 256 × 256 → shape `(128, 256, 256, 1)`
- A batch of 128 color images → shape `(128, 256, 256, 3)`

There are two conventions for image tensor shapes:
- **Channels-last** (standard in TensorFlow, Keras): `(samples, height, width, color_depth)`
- **Channels-first** (standard in PyTorch): `(samples, color_depth, height, width)`

**Video data:**
- A 60-second, 144 × 256 YouTube clip at 4 fps (240 frames), batch of 4 → shape `(4, 240, 144, 256, 3)` — over 106 million values!

---

## 2.3 The Gears of Neural Networks: Tensor Operations

Just like any computer program can be ultimately reduced to a small set of binary operations on binary inputs, all transformations learned by deep neural networks can be reduced to a handful of **tensor operations** applied to tensors of numeric data.

A Keras layer instance like:

```python
keras.layers.Dense(512, activation="relu")
```

can be interpreted as a function implementing:

```python
output = relu(matmul(input, W) + b)
```

We have three tensor operations here:

1. A **tensor product** (`matmul`) between the input tensor and a tensor named `W`
2. An **addition** (`+`) between the resulting matrix and a vector `b`
3. A **relu** operation: `relu(x) is max(x, 0)`

### 2.3.1 Element-Wise Operations

The relu operation and addition are **element-wise operations**: operations applied independently to each entry in the tensors. This makes them highly amenable to massively parallel implementations.

```python
# Naive Python implementation of element-wise relu:
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

# Naive element-wise addition:
def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x
```

In practice, NumPy uses highly optimized BLAS (Basic Linear Algebra Subprograms) implementations, which are blazing fast:

```python
import numpy as np
z = x + y
z = np.maximum(z, 0.0)
```

Timing comparison: vectorized NumPy takes ~0.02s vs. ~2.45s for the naive loop version over 1,000 iterations.

### 2.3.2 Broadcasting

When adding tensors of different shapes, the smaller tensor will be **broadcast** to match the shape of the larger tensor. Broadcasting consists of two steps:

1. Axes (called broadcast axes) are added to the smaller tensor to match the `ndim` of the larger tensor.
2. The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor.

```python
import numpy as np

X = np.random.random((32, 10))  # shape (32, 10)
y = np.random.random((10,))     # shape (10,)

# y is broadcast to (32, 10) — virtually, not actually in memory
# This works with broadcasting:
z = np.maximum(x, y)  # x shape (64, 3, 32, 10), y shape (32, 10)
```

### 2.3.3 Tensor Product

The **tensor product** (also called dot product or `matmul`) is one of the most common and useful tensor operations.

```python
x = np.random.random((32,))
y = np.random.random((32,))

z = np.matmul(x, y)
z = x @ y  # shorthand
```

Key rules:
- The product of two vectors produces a **scalar** (vectors must have the same number of elements)
- The product of a matrix `x` and vector `y` produces a vector
- You can take the product of two matrices `x` and `y` if and only if `x.shape[1] == y.shape[0]`, producing a matrix of shape `(x.shape[0], y.shape[1])`
- Once one tensor has `ndim > 1`, matmul is no longer symmetric: `matmul(x, y) != matmul(y, x)`

More generally:
```
(a, b, c, d) • (d,)    -> (a, b, c)
(a, b, c, d) • (d, e)  -> (a, b, c, e)
```

### 2.3.4 Tensor Reshaping

**Reshaping** a tensor means rearranging its rows and columns to match a target shape. The reshaped tensor has the same total number of coefficients as the initial tensor.

```python
>>> x = np.array([[0., 1.],
...               [2., 3.],
...               [4., 5.]])
>>> x.shape
(3, 2)
>>> x = x.reshape((6, 1))
>>> x = x.reshape((2, 3))

# Transposition: exchanging rows and columns
>>> x = np.zeros((300, 20))
>>> x = np.transpose(x)
>>> x.shape
(20, 300)
```

### 2.3.5 Geometric Interpretation of Tensor Operations

Because the contents of tensors can be interpreted as coordinates of points in some geometric space, all tensor operations have a **geometric interpretation**.

Elementary geometric operations expressible as tensor operations:

- **Translation** — Adding a vector moves a point by a fixed amount in a fixed direction
- **Rotation** — A counterclockwise rotation by angle theta uses the matrix `R = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]`
- **Scaling** — A diagonal matrix achieves vertical and horizontal scaling
- **Linear transform** — A product with an arbitrary matrix (includes scaling and rotation)
- **Affine transform** — The combination of a linear transform and a translation: `y = W @ x + b` — exactly what a `Dense` layer without activation implements

> **Important:** If you apply many affine transforms repeatedly, you still end up with an affine transform. This is why we need activation functions like relu. Thanks to activation functions, a chain of Dense layers can implement very complex, **nonlinear** geometric transformations. Without them, a deep neural network would just be a linear model in disguise!

### 2.3.6 A Geometric Interpretation of Deep Learning

You can interpret a neural network as a very complex geometric transformation in a high-dimensional space, implemented via a series of simple steps.

A useful 3D mental image: imagine two sheets of colored paper (one red, one blue) crumpled together into a ball. That crumpled ball is your input data, and each sheet is a class. What a neural network does is find a transformation that would uncrumple the ball to make the two classes cleanly separable.

Deep learning takes the approach of incrementally decomposing a complicated geometric transformation into a long chain of elementary ones. Each layer applies a transformation that disentangles the data a little—and a deep stack of layers makes tractable an extremely complicated disentanglement process.

---

## 2.4 The Engine of Neural Networks: Gradient-Based Optimization

Each neural layer transforms its input data as follows:

```python
output = relu(matmul(input, W) + b)
```

In this expression, `W` and `b` are tensors called the **weights** or **trainable parameters** of the layer (the `kernel` and `bias` attributes, respectively). These weights contain the information learned by the model from exposure to training data.

Initially, weight matrices are filled with small random values (**random initialization**). Training works within a **training loop**:

1. Draw a batch of training samples `x` and corresponding targets `y_true`
2. Run the model on `x` (the **forward pass**) to obtain predictions `y_pred`
3. Compute the **loss** of the model on the batch — a measure of the mismatch between `y_pred` and `y_true`
4. Update all weights of the model in a way that slightly reduces the loss on this batch

### 2.4.1 What's a Derivative?

Consider a continuous, smooth function `f(x) = y`. Because the function is smooth, when `epsilon_x` is small enough around a certain point `p`, it's possible to approximate `f` as a linear function of slope `a`:

```
f(x + epsilon_x) = y + a * epsilon_x
```

The slope `a` is called the **derivative** of `f` in `p`. If `a` is negative, a small increase in `x` will result in a decrease of `f(x)`. If `a` is positive, a small increase in `x` will result in an increase of `f(x)`.

To minimize `f(x)`, you just need to move `x` a little in the **opposite direction** from the derivative.

### 2.4.2 Derivative of a Tensor Operation: The Gradient

The derivative of a tensor operation (or tensor function) is called a **gradient**. Gradients are the generalization of derivatives to functions that take tensors as inputs.

Given:
- Input vector `x` (a sample)
- Matrix `W` (model weights)
- Target `y_true`
- Loss function `loss`

We can write `loss_value = f(W)`. The gradient `grad(loss_value, W0)` is a tensor of the same shape as `W`, where each coefficient indicates the direction and magnitude of the change in `loss_value` when modifying the corresponding weight.

To reduce the loss, move `W` in the **opposite direction from the gradient**:

```
W1 = W0 - step * grad(f(W0), W0)
```

The scaling factor `step` is needed because the gradient only approximates the curvature near `W0`.

### 2.4.3 Stochastic Gradient Descent

The algorithm called **mini-batch stochastic gradient descent (mini-batch SGD)**:

1. Draw a batch of training samples `x` and corresponding targets `y_true`
2. Run the model on `x` to obtain predictions `y_pred` (forward pass)
3. Compute the loss of the model on the batch
4. Compute the **gradient** of the loss with regard to the model's parameters (backward pass)
5. Move the parameters a little in the opposite direction from the gradient:
   ```
   W -= learning_rate * gradient
   ```

The **learning rate** modulates the "speed" of gradient descent. If it's too small, descent takes many iterations and can get stuck. If too large, updates may be completely random.

Variants:
- **True SGD** — draws a single sample per iteration
- **Batch gradient descent** — runs on all available data each step
- **Mini-batch SGD** — the efficient compromise (most common)

**SGD variants (optimizers):** SGD with momentum, Adagrad, RMSprop, Adam, and others. **Momentum** addresses convergence speed and local minima by moving the ball based not only on the current slope but also on current velocity (past acceleration):

```python
past_velocity = 0.0
momentum = 0.1
while loss > 0.01:
    w, loss, gradient = get_current_parameters()
    velocity = past_velocity * momentum - learning_rate * gradient
    w = w + momentum * velocity - learning_rate * gradient
    past_velocity = velocity
    update_parameter(w)
```

### 2.4.4 Chaining Derivatives: The Backpropagation Algorithm

**Backpropagation** is a way to use the derivative of simple operations (addition, relu, tensor product) to compute the gradient of arbitrarily complex combinations of these atomic operations.

#### The Chain Rule

For two functions `f` and `g`, and the composed function `fg` such that `y = f(g(x))`:

```python
grad(y, x) == grad(y, x1) * grad(x1, x)
```

For a longer chain:

```python
def fghj(x):
    x1 = j(x)
    x2 = h(x1)
    x3 = g(x2)
    y = f(x3)
    return y

grad(y, x) == grad(y, x3) * grad(x3, x2) * grad(x2, x1) * grad(x1, x)
```

#### Automatic Differentiation with Computation Graphs

A **computation graph** is a directed acyclic graph of tensor operations. Backpropagation is simply the application of the chain rule to a computation graph.

In TensorFlow, the `tf.GradientTape` object records tensor operations and allows retrieving gradients:

```python
import tensorflow as tf

x = tf.zeros(shape=())
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)
```

Modern frameworks (JAX, TensorFlow, PyTorch) implement **automatic differentiation**, making it possible to retrieve gradients of arbitrary compositions of differentiable tensor operations without writing them by hand.

---

## 2.5 Looking Back at Our First Example

Let's go back to the first example and review each piece in the light of what we've learned.

**The input data:**

```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```

The input images are stored in `float32` tensors of shape `(60000, 784)` (training) and `(10000, 784)` (test).

**The model:**

```python
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
```

This model consists of a chain of two Dense layers, each applying simple tensor operations involving weight tensors. Weight tensors are where the knowledge of the model persists.

**The compilation step:**

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

`sparse_categorical_crossentropy` is the loss function used as feedback signal. The reduction of the loss happens via mini-batch stochastic gradient descent, governed by the `adam` optimizer.

**The training loop:**

```python
model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=128,
)
```

The model iterates on the training data in mini-batches of 128 samples, 5 times over (each iteration over all training data is called an **epoch**). For each batch, the model computes the gradient of the loss with regard to the weights via backpropagation and moves the weights to reduce the loss.

After 5 epochs, the model will have performed 2,345 gradient updates (469 per epoch).

### 2.5.1 Reimplementing Our First Example from Scratch

**A simple Dense class:**

```python
import keras
from keras import ops

class NaiveDense:
    def __init__(self, input_size, output_size, activation=None):
        self.activation = activation
        self.W = keras.Variable(
            shape=(input_size, output_size), initializer="uniform"
        )
        self.b = keras.Variable(shape=(output_size,), initializer="zeros")

    def __call__(self, inputs):
        x = ops.matmul(inputs, self.W)
        x = x + self.b
        if self.activation is not None:
            x = self.activation(x)
        return x

    @property
    def weights(self):
        return [self.W, self.b]
```

**A simple Sequential class:**

```python
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
```

**Creating a mock Keras model:**

```python
model = NaiveSequential(
    [
        NaiveDense(input_size=28 * 28, output_size=512, activation=ops.relu),
        NaiveDense(input_size=512, output_size=10, activation=ops.softmax),
    ]
)
assert len(model.weights) == 4
```

**A batch generator:**

```python
import math

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels
```

### 2.5.2 Running One Training Step

**Listing 2.9 — A single step of training:**

```python
def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        loss = ops.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = ops.mean(loss)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss
```

**The weight update step:**

```python
learning_rate = 1e-3

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign(w - g * learning_rate)

# Or using a Keras optimizer:
from keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))
```

### 2.5.3 The Full Training Loop

```python
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")
```

```python
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128)
```

### 2.5.4 Evaluating the Model

```python
>>> predictions = model(test_images)
>>> predicted_labels = ops.argmax(predictions, axis=1)
>>> matches = predicted_labels == test_labels
>>> f"accuracy: {ops.mean(matches):.2f}"
accuracy: 0.83
```

---

## Summary

- **Tensors** form the foundation of modern machine learning systems. They come in various flavors of `dtype`, rank, and shape.
- You can manipulate numerical tensors via **tensor operations** (addition, tensor product, element-wise multiplication), which can be interpreted as encoding geometric transformations. Everything in deep learning is amenable to a geometric interpretation.
- Deep learning models consist of chains of simple tensor operations, **parameterized by weights**. The weights of a model are where its "knowledge" is stored.
- **Learning** means finding a set of values for the model's weights that minimizes a loss function for a given set of training data samples and their corresponding targets.
- Learning happens by drawing random batches of data and computing the **gradient** of the model parameters with respect to the loss on the batch. The parameters are then moved a bit in the **opposite direction** from the gradient. This is called **mini-batch gradient descent**.
- The entire learning process is made possible by the fact that all tensor operations in neural networks are **differentiable**, allowing the **chain rule** to find the gradient function — this is called **backpropagation**.
- Two key concepts you'll see frequently going forward:
  - The **loss** — the quantity you'll attempt to minimize during training, representing a measure of success for the task you're trying to solve.
  - The **optimizer** — specifies how the gradient of the loss will be used to update parameters (e.g., RMSProp, SGD with momentum, Adam).
