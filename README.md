# SparksNet - A Deep Learning Framework

SparksNet is a lightweight deep learning framework built from scratch to help you learn the core principles of neural networks and machine learning. It implements key components like activations, layers, loss functions, optimizers, and more, with a focus on simplicity and extensibility.

## Project Structure

```
My Deep Learning Framework (SparksNet)/
│   ├── generate_project_structure.py        # Script to generate project structure
│   ├── project_structure.txt               # Text file with details of project structure
│   ├── example_1 (Regression)/             # Example for regression
│   │   ├── gradient_descent.py
│   │   ├── model.py
│   │   ├── stochastic_gradient_descent.py
│   ├── example_2 (Classification)/         # Example for classification
│   │   ├── model.py
│   │   ├── test.csv
│   │   ├── train.csv
│   ├── SparksNet/                          # Core framework code
│   │   ├── __init__.py
│   │   ├── activations/                   # Activation functions
│   │   │   ├── relu.py
│   │   │   ├── sigmoid.py
│   │   │   ├── softmax.py
│   │   │   ├── tanh.py
│   │   │   ├── __init__.py
│   │   ├── core/                          # Core modules for building models
│   │   │   ├── sequential.py
│   │   │   ├── __init__.py
│   │   ├── layers/                        # Layer definitions
│   │   │   ├── linearlayer.py
│   │   │   ├── __init__.py
│   │   ├── loss/                          # Loss functions
│   │   │   ├── bce_loss.py
│   │   │   ├── bce_with_logit_loss.py
│   │   │   ├── cross_entropy_loss.py
│   │   │   ├── l1_loss.py
│   │   │   ├── mse_loss.py
│   │   │   ├── __init__.py
│   │   ├── optim/                         # Optimizers
│   │   │   ├── adam.py
│   │   │   ├── momentum.py
│   │   │   ├── rms_prop.py
│   │   │   ├── sgd.py
│   │   │   ├── __init__.py
```

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/Himanshu7921/SparksNet
```

2. Install the necessary dependencies:

```bash
pip install numpy
pip install sklearn
```

## Framework Overview

### Key Components:

- **Activations**: Implements ReLU, Sigmoid, Softmax, Tanh.
- **Layers**: Supports `Linear` layer to build neural networks.
- **Loss**: Implements multiple loss functions like MSE, Cross-Entropy, BCE, etc.
- **Optimizers**: Includes basic optimizers like SGD, Adam, RMSProp.

### Sequential API:

`SparksNet` provides a `Sequential` class to easily stack layers and define a model. You can add layers like Linear, ReLU, Tanh, and other activation functions to construct a deep learning model.

### Example Code Snippets

#### Example 1: Regression (Linear Regression)

```python
import sys
import os
import numpy as np
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Framework specific imports
import SparksNet
from SparksNet.layers import Linear
from SparksNet.activations import Tanh
from SparksNet.loss import MSELoss
from SparksNet.core import Sequential

class MyModel:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.seq = Sequential(
            Linear(input_size=in_features, output_size=64),
            Tanh(),
            Linear(input_size=64, output_size=64),
            Tanh(),
            Linear(input_size=64, output_size=out_features)
        )

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.seq(X)

    def backward(self, grad):
        self.seq.backward(grad)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def parameters(self):
        return self.seq.parameters()

# Initialize loss function and optimizer
loss_fn = MSELoss()
X = np.random.rand(2050, 1)
Y = 3 * X + 5  # y = 3x + 5 (linear regression example)

train_split = int(len(X) * 0.8)
X_train, y_train = X[:train_split], Y[:train_split]
X_test, y_test = X[train_split:], Y[train_split:]

model = MyModel(X.shape[1], 1)
optimizer = SparksNet.SGD(model.parameters(), lr=0.01, gamma=0.9)

# Training Loop
epochs = 2500
for epoch in range(epochs):
    y_preds = model(X_train)
    loss = loss_fn(y_preds, y_train)
    grad_loss = loss_fn.backward()
    model.backward(grad_loss)
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch: {epoch} | Loss: {loss}")

# Evaluate model on test data
y_test_preds = model(X_test)
test_loss = loss_fn(y_test_preds, y_test)
print(f"Testing Loss: {test_loss}")

# Regression evaluation with MAE
print("MAE:", mean_absolute_error(y_test, y_test_preds))
```

**Training Output Example**:

```
Epoch: 100/2500 | Loss: 0.06873848247621167
Epoch: 2500/2500 | Loss: 3.293192561092406e-06
Testing Loss: 0.14126645815291203
MAE: 0.382
```

#### Example 2: Classification (MNIST)

```python
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Framework specific imports
import SparksNet
from SparksNet.layers import Linear
from SparksNet.activations import ReLU, Softmax
from SparksNet.loss import CrossEntropyLoss
from SparksNet.core import Sequential

class MyModel:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.seq = Sequential(
            Linear(input_size=in_features, output_size=128),
            ReLU(),
            Linear(input_size=128, output_size=64),
            ReLU(),
            Linear(input_size=64, output_size=out_features)  # 10 output classes for MNIST
        )

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.seq(X)

    def backward(self, grad):
        self.seq.backward(grad)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def parameters(self):
        return self.seq.parameters()

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype(np.float32) / 255
x_test = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255

# Initialize the model
model = MyModel(in_features=28*28, out_features=10)  # 10 classes for MNIST
optimizer = SparksNet.SGD(model.parameters(), lr=0.001, gamma=0.9)
loss_fn = CrossEntropyLoss()

# Training loop
epochs = 200
batch_size = 32
interval = 100
for epoch in range(epochs):
    permutation = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[permutation]
    y_train_shuffled = y_train[permutation]

    # Mini-batch gradient descent
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        y_preds = model(x_batch)
        loss = loss_fn(y_preds, y_batch)

        grad_loss = loss_fn.backward()
        model.backward(grad_loss)
        optimizer.step()

    if epoch % interval == 0:
        print(f"Epoch: {epoch}/{epochs} | Loss: {loss}")

# Evaluate model on test data
y_test_preds = model(x_test)
test_loss = loss_fn(y_test_preds, y_test)
print(f"Testing Loss: {test_loss}")

# Convert logits to predicted classes
y_test_preds_classes = np.argmax(y_test_preds, axis=1)

# Evaluate classification accuracy
accuracy = accuracy_score(y_test, y_test_preds_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

**Training Output Example**:

```
Epoch: 100/200 | Loss: 0.06873848247621167
Epoch: 200/200 | Loss: 3.293192561092406e-06
Testing Loss: 0.14126645815291203
Accuracy: 98.33%
Model's predictions (first 10): [7 2 1 0 4 1 9 5 9 1]
Actual Labels (first 10): [7 2 1 0 4 1 9 5 9 1]
```

## Contributing

Feel free to fork this repository, submit issues, and send pull requests to improve the framework.

## License

MIT License
