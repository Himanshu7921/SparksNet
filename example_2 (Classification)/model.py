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
        """
        Initialize the model with the required layers.
        - 2 hidden layers with ReLU activations
        - Output layer with Softmax activation for multi-class classification
        """
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
x_train = x_train.reshape(-1, 28 * 28).astype(np.float32) / 255  # Flatten and normalize
x_test = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255  # Flatten and normalize

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
        print(f"Epoch: {epoch + interval}/{epochs} | Loss: {loss}")

# Evaluate model on test data
y_test_preds = model(x_test)
test_loss = loss_fn(y_test_preds, y_test)
print(f"Testing Loss: {test_loss}")

# Convert logits to predicted classes
y_test_preds_classes = np.argmax(y_test_preds, axis=1)

# Evaluate classification accuracy
accuracy = accuracy_score(y_test, y_test_preds_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display some predictions vs actual values
print(f"Model's Predictions (first 10): {y_test_preds_classes[:10]}")
print(f"Actual Labels (first 10): {y_test[:10]}")
