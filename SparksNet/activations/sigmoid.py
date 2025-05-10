import numpy as np

class Sigmoid:
    """
    Sigmoid Activation Function.

    Applies the element-wise function:
        Sigmoid(x) = 1 / (1 + exp(-x))

    During the backward pass, the derivative is:
        Sigmoid'(x) = Sigmoid(x) * (1 - Sigmoid(x))

    This squashes input values to the range (0, 1), useful for binary classification and
    final layer activations in simple models.

    Example:
        >>> sigmoid = Sigmoid()
        >>> out = sigmoid(X)             # Forward pass
        >>> grad_in = sigmoid.backward(dout)  # Backward pass
    """

    def __init__(self) -> None:
        """
        Initializes the Sigmoid activation.
        """
        self.input = None
        self.output = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the Sigmoid activation function element-wise.

        Args:
            X (np.ndarray): Input array.

        Returns:
            np.ndarray: Activated output in the range (0, 1).
        """
        self.input = X
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss w.r.t input for backpropagation.

        Args:
            grad_output (np.ndarray): Gradient flowing from the next layer.

        Returns:
            np.ndarray: Gradient after applying the derivative of Sigmoid.
        """
        sigmoid_grad = self.output * (1 - self.output)
        return sigmoid_grad * grad_output

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Enables instance to be used as a function for the forward pass.

        Args:
            X (np.ndarray): Input array.

        Returns:
            np.ndarray: Activated output.
        """
        return self.forward(X)
