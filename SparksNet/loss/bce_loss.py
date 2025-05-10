import numpy as np

class BCELoss:
    """
    Binary Cross-Entropy Loss (BCELoss) is a loss function used for binary classification tasks.
    It is used when the model output has already been passed through a Sigmoid activation function.
    
    BCELoss computes the dissimilarity between the predicted probabilities and the actual binary labels.
    
    Mathematically:
        BCELoss(ŷ, y) = -[y * log(ŷ) + (1 - y) * log(1 - ŷ)]
    
    where:
        - ŷ = predicted probability (after sigmoid activation), 0 < ŷ < 1
        - y  = ground truth labels (0 or 1)
    
    Use Case:
        - This loss should be applied when your model outputs are already passed through a Sigmoid function.
        - For raw logits (before sigmoid activation), use `BCEWithLogitsLoss` for better numerical stability.

    Numerical Stability:
        - To avoid log(0), a small epsilon (e.g., 1e-9) is added during log operations.
    
    Derivative of BCELoss with respect to ŷ (prediction):
        dL/dŷ = -y / ŷ + (1 - y) / (1 - ŷ)
    
    Example:
        >>> bce_loss = BCELoss()
        >>> loss = bce_loss.forward(y_pred, target)
        >>> grad = bce_loss.backward(grad_output)
    """
    
    def __init__(self) -> None:
        """
        Initializes the Binary Cross-Entropy Loss function.
        Should be used when predictions are already passed through a Sigmoid activation.
        """
        self.y_preds: np.ndarray = None
        self.target: np.ndarray = None
        self.N: int = None

    def forward(self, y_preds: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the Binary Cross-Entropy loss.

        Args:
            y_preds (np.ndarray): Sigmoid-activated predictions (0 < ŷ < 1).
            target (np.ndarray): Ground truth labels (0 or 1).

        Returns:
            float: Scalar binary cross-entropy loss.
        """
        self.y_preds = y_preds
        self.target = target
        self.N = self.target.shape[0]  # Extract batch size

        return -np.mean(
            (self.target * np.log(self.y_preds + 1e-9)) + (1 - self.target) * np.log(1 - self.y_preds + 1e-9)
        )

    def backward(self, grad_output: float = 1.0) -> np.ndarray:
        """
        Computes the gradient of the BCE loss with respect to the predictions.

        Args:
            grad_output (float): Incoming gradient from the next layer (default is 1.0).

        Returns:
            np.ndarray: Gradient of loss with respect to the predictions.
        """
        grad = -(self.target / (self.y_preds + 1e-9)) + ((1 - self.target) / (1 - self.y_preds + 1e-9))
        return grad_output * grad / self.N  # Normalize by batch size (optional)

    def __call__(self, y_pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Enables instance to be used as a function for the forward pass.

        Args:
            y_pred (np.ndarray): Sigmoid-activated predictions (0 < ŷ < 1).
            target (np.ndarray): Ground truth labels (0 or 1).

        Returns:
            float: Scalar binary cross-entropy loss.
        """
        return self.forward(y_pred, target)
