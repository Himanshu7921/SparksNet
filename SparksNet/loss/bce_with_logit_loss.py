import numpy as np
from SparksNet.activations import Sigmoid

class BCEWithLogitsLoss:
    """
    Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss) combines the Sigmoid activation 
    and Binary Cross-Entropy loss into a single, more numerically stable operation.
    
    BCEWithLogitsLoss is used to compute binary classification losses when the model output is 
    in the form of raw logits (before Sigmoid activation).
    
    Mathematically:
        BCEWithLogitsLoss(z, y) = -[y * log(sigmoid(z)) + (1 - y) * log(1 - sigmoid(z))]
    
    where:
        - z = raw logits (output from the model)
        - y = ground truth labels
        - sigmoid(z) = Sigmoid function applied to logits

    Why fused?
        Fusing the Sigmoid activation and BCE loss improves numerical stability by avoiding separate 
        sigmoid and log operations during the forward pass.
    
    Derivative of BCEWithLogitsLoss with respect to logits:
        dL/dz = sigmoid(z) - y
    
    This simplified derivative avoids the need for computing separate gradients of the Sigmoid 
    function and BCE loss, making the backward pass both efficient and stable.

    Example:
        >>> bce_logits_loss = BCEWithLogitsLoss()
        >>> loss = bce_logits_loss.forward(logits, target)
        >>> grad = bce_logits_loss.backward(grad_output)
    """
    
    def __init__(self) -> None:
        """
        Initializes the BCEWithLogitsLoss class.
        Sets up the Sigmoid activation and stores placeholders for model predictions 
        and target values to be used during the backward pass.
        """
        self.sigmoid: Sigmoid = Sigmoid()
        self.y_preds: np.ndarray = None
        self.target: np.ndarray = None
        self.N: int = None

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        """
        Applies the Sigmoid activation function to logits and computes the Binary Cross-Entropy loss.

        Args:
            logits (np.ndarray): Raw logits (model output before applying the Sigmoid activation).
            target (np.ndarray): Ground truth labels (0 or 1), same shape as logits.

        Returns:
            float: Computed Binary Cross-Entropy loss with logits.
        """
        self.y_preds = self.sigmoid.forward(logits)
        self.target = target
        self.N = self.target.shape[0]  # Extract batch size

        return - np.mean(
            (self.target * np.log(self.y_preds + 1e-9)) + (1 - self.target) * np.log(1 - self.y_preds + 1e-9)
        )

    def backward(self, grad_output: float = 1.0) -> np.ndarray:
        """
        Backward pass for the BCEWithLogitsLoss.
        Computes the gradient of the loss with respect to the raw logits before applying the Sigmoid.

        Uses the identity:
            dL/dz = sigmoid(z) - y

        Args:
            grad_output (float): Gradient from the next layer (default is 1.0).

        Returns:
            np.ndarray: Gradient of the loss with respect to the logits.
        """
        return (self.y_preds - self.target) * grad_output
    
    def __call__(self, logits: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Allows the BCEWithLogitsLoss class to be called like a function.
        Computes the Binary Cross-Entropy loss for the provided logits and targets.

        Args:
            logits (np.ndarray): Raw logits (model output before applying the Sigmoid activation).
            target (np.ndarray): Ground truth labels (0 or 1).

        Returns:
            float: Computed Binary Cross-Entropy loss.
        """
        return self.forward(logits, target)
