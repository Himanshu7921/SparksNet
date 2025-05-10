import numpy as np

class Tanh:
    """
    Tanh Activation Function.

    The tanh (hyperbolic tangent) function is a smooth, non-linear activation function.
    It outputs values in the range (-1, 1), making it centered at zero. This property is beneficial
    for faster convergence in neural networks compared to the Sigmoid function.

    Tanh Formula:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    The derivative of tanh is:
        d(tanh(x))/dx = 1 - tanh(x)^2

    Example:
        >>> tanh = Tanh()
        >>> output = tanh(X)         # Forward pass
        >>> grad_input = tanh.backward(grad_output)  # Backward pass
    """
    
    def __init__(self) -> None:
        """
        Initializes the Tanh activation function.
        """
        self.input: np.ndarray = None
        self.output: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass of the tanh activation function.

        Args:
            X (np.ndarray): Input array of any shape.

        Returns:
            np.ndarray: Output after applying tanh activation.
        """
        self.input = X
        exp_minus = np.exp(-X)
        exp_plus = np.exp(X)
        self.output = (exp_plus - exp_minus) / (exp_plus + exp_minus)
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of the tanh activation function.
        Computes the gradient of the loss with respect to the input.

        Args:
            grad_output (np.ndarray): Gradient flowing from the next layer.

        Returns:
            np.ndarray: Gradient with respect to the input.
        """
        return grad_output * (1 - self.output ** 2)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Enables instance to be used as a function for the forward pass.

        Args:
            X (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying tanh activation.
        """
        return self.forward(X)
