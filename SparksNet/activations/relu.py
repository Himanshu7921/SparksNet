import numpy as np


class ReLU:
    """
    ReLU (Rectified Linear Unit) Activation Function.

    Applies the element-wise function:
        ReLU(x) = max(0, x)

    During the backward pass, the derivative is:
        ReLU'(x) = 1 if x > 0 else 0

    This helps introduce non-linearity into the network and prevents vanishing gradients 
    for positive values.

    Example:
        >>> relu = ReLU()
        >>> out = relu(X)             # Forward pass
        >>> grad_in = relu.backward(dout)  # Backward pass
    """

    def __init__(self) -> None:
        """
        Initializes the ReLU activation.
        """
        self.input = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Applies ReLU activation function element-wise.

        Args:
            X (np.ndarray): Input array.

        Returns:
            np.ndarray: Activated output.
        """
        self.input = X
        return np.maximum(0, X)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes gradient of the loss w.r.t input for backpropagation.

        Args:
            grad_output (np.ndarray): Gradient flowing from the next layer.

        Returns:
            np.ndarray: Modified gradient after applying ReLU derivative.
        """
        relu_grad = self.input > 0
        return relu_grad * grad_output

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Enables instance to be used as a function for the forward pass.

        Args:
            X (np.ndarray): Input array.

        Returns:
            np.ndarray: Activated output.
        """
        return self.forward(X)
