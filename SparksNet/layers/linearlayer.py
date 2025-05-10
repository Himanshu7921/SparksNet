import numpy as np

class Linear:
    """
    A fully connected (dense) layer that performs an affine transformation.

    Applies a linear transformation to the incoming data: 
        Y = X @ W + b

    where:
        - X is the input
        - W is the weight matrix
        - b is the bias vector

    Attributes:
        weight (np.ndarray): Learnable weights of shape (input_size, output_size)
        bias (np.ndarray): Learnable biases of shape (output_size,)
        grad_weight (np.ndarray): Gradient of the loss w.r.t. weights
        grad_bias (np.ndarray): Gradient of the loss w.r.t. biases

    Example:
        >>> fc = Linear(784, 128)
        >>> out = fc(X)              # Forward pass
        >>> grad_in = fc.backward(dout)  # Backward pass
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initializes the Linear layer with random weights and biases.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        self.input_size = input_size
        self.output_size = output_size

        self.weight = np.random.rand(input_size, output_size) * 0.01
        self.bias = np.random.rand(output_size)

        self.input = None
        self.grad_weight = None
        self.grad_bias = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass.

        Args:
            X (np.ndarray): Input of shape (batch_size, input_size) or (input_size,)

        Returns:
            np.ndarray: Output of shape (batch_size, output_size)
        """
        self.__check_input(X)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        self.input = X
        return X @ self.weight + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes gradients for backward pass.

        Args:
            grad_output (np.ndarray): Gradient from the next layer.

        Returns:
            np.ndarray: Gradient to be passed to the previous layer.
        """
        if grad_output.ndim == 1:
            grad_output = grad_output[np.newaxis, :]
        if self.input.ndim == 1:
            self.input = self.input[np.newaxis, :]

        self.grad_weight = self.input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        
        return grad_output @ self.weight.T

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Enables the instance to be called as a function for the forward pass.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Output after transformation.
        """
        return self.forward(X)

    def __check_input(self, X: np.ndarray) -> None:
        """
        Validates input shape.

        Args:
            X (np.ndarray): Input data.

        Raises:
            TypeError: If input is not a NumPy array.
            ValueError: If input shape does not match layer configuration.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Expected input to be np.ndarray")
        if X.ndim == 1:
            if X.shape[0] != self.input_size:
                raise ValueError(f"Expected input shape ({self.input_size},), got {X.shape}")
        elif X.ndim == 2:
            if X.shape[1] != self.input_size:
                raise ValueError(f"Expected input shape (, {self.input_size}), got {X.shape}")
        else:
            raise ValueError("Input must be 1D or 2D array")
