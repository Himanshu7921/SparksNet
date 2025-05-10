import numpy as np

class Softmax:
    """
    Softmax Activation Function.

    The Softmax function converts logits into probabilities by exponentiating each logit
    and normalizing by the sum of all exponentiated logits.

    Softmax Formula:
        Softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)
    
    Where:
    - z_i is the i-th input logit.
    - The sum is over all logits for the class.

    Outputs values between 0 and 1 that sum to 1, representing a probability distribution.
    Used primarily in multi-class classification tasks, particularly for the final layer.

    Softmax Derivative Formula (Jacobian):
        The derivative of Softmax w.r.t. its inputs is represented by the Jacobian matrix:
        
        - d(Softmax(z_i)) / dz_k = -Softmax(z_i) * Softmax(z_k) if i ≠ k
        - d(Softmax(z_i)) / dz_k = Softmax(z_i) * (1 - Softmax(z_i)) if i = k
        
    Example:
        >>> softmax = Softmax()
        >>> out = softmax(X)         # Forward pass
        >>> grad_in = softmax.backward(dout)  # Backward pass
    """
    
    def __init__(self) -> None:
        """
        Initializes the Softmax activation function.
        """
        self.input: np.ndarray = None
        self.output: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the Softmax activation function to the input logits.

        Args:
            X (np.ndarray): Logits for the input data, typically of shape (batch_size, num_classes).

        Returns:
            np.ndarray: Softmax probabilities, shape (batch_size, num_classes).
        """
        self.input = X
        exps = np.exp(self.input - np.max(self.input, axis=1, keepdims=True))  # For numerical stability
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss w.r.t input logits during the backward pass.

        Args:
            grad_output (np.ndarray): Gradient from the next layer, shape (batch_size, num_classes).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input logits, shape (batch_size, num_classes).
        """
        grad_input = np.zeros_like(self.output)
        batch_size, _ = self.output.shape

        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)  # shape: (num_classes, 1)
            jacobian = np.diagflat(y) - y @ y.T  # Jacobian matrix (num_classes, num_classes)
            grad_input[i] = jacobian @ grad_output[i]  # shape: (num_classes,)
        
        return grad_input

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Enables instance to be used as a function for the forward pass.

        Args:
            X (np.ndarray): Input logits.

        Returns:
            np.ndarray: Softmax probabilities.
        """
        return self.forward(X)
