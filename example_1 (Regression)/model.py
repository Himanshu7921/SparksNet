import sys
import os
import numpy as np
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Framework specific imports
import SparksNet
from SparksNet.layers import Linear
from SparksNet.activations import ReLU, Softmax, Tanh
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
X = np.random.rand(2050, 1)  # 2050 samples, each with 1 input feature
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

    # Log training progress every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch: {epoch} | Loss: {loss}")

# Evaluate model on test data
y_test_preds = model(X_test)
test_loss = loss_fn(y_test_preds, y_test)
print(f"Testing Loss: {test_loss}")

# Regression evaluation with MAE
print("MAE:", mean_absolute_error(y_test, y_test_preds))

# Display some predictions vs actual values
print(f"Model's Predictions (first 10): {y_test_preds[:10]}")
print(f"Actual Labels (first 10): {y_test[:10]}")
