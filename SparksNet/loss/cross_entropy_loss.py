import numpy as np
from SparksNet.activations import Softmax

class CrossEntropyLoss:
    """
    Cross-Entropy Loss for multi-class classification problems. This loss function calculates the 
    difference between the predicted class probabilities and the actual class labels.
    
    CrossEntropyLoss is commonly used for multi-class classification where the model output is 
    raw logits, which are converted into class probabilities using the Softmax function.

    Definition:
        CrossEntropyLoss = -1/N * sum(log(softmax(logits)[i]) for i in range(N) if target[i] == 1)
    
    where:
        - logits = raw model output (before Softmax)
        - softmax(logits) = probability distribution over all classes
        - target = integer class labels (not one-hot encoded)
    
    Derivative:
        dL/d(logits) = softmax(logits) - target
    
    This derivative simplifies the computation of the gradient for backpropagation, making it 
    efficient for large-scale multi-class classification problems.
    """

    def __init__(self):
        """
        Initializes the CrossEntropyLoss class.
        Sets up the Softmax activation and stores placeholders for model predictions 
        and target values to be used during the backward pass.
        """
        self.softmax: Softmax = Softmax()
        self.y_pred: np.ndarray = None
        self.target: np.ndarray = None
        self.N: int = None

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        """
        Forward pass of the CrossEntropyLoss. Applies Softmax to logits and computes the Cross-Entropy loss.

        Args:
            logits (np.ndarray): Raw logits output from the model (before applying Softmax).
            target (np.ndarray): Ground truth labels (integer class indices).

        Returns:
            float: Computed Cross-Entropy loss value.
        """
        self.y_pred = self.softmax.forward(logits)
        self.N = logits.shape[0]  # Extract batch size
        self.target = target
        
        # Compute the log loss for the correct classes (given as integer indices)
        correct_class_probs = self.y_pred[np.arange(self.N), self.target]  # Get probs for correct classes
        loss = -np.mean(np.log(correct_class_probs + 1e-9))  # Adding epsilon to avoid log(0)
        
        return loss

    def backward(self, grad_output: float = 1.0) -> np.ndarray:
        """
        Backward pass of the CrossEntropyLoss. Computes the gradient of the loss with respect to the logits.
        Uses the derivative: dL/d(logits) = softmax(logits) - target

        Args:
            grad_output (float): Gradient from the next layer (default is 1.0).

        Returns:
            np.ndarray: Gradient of the loss with respect to the logits (before applying Softmax).
        """
        # Expand target to match the shape of self.y_pred for broadcasting
        target_one_hot = np.zeros_like(self.y_pred)  # Create a zero matrix with the same shape as self.y_pred
        target_one_hot[np.arange(self.N), self.target] = 1  # Set the correct class positions to 1

        # The gradient of the cross-entropy loss with respect to logits is the softmax output minus the one-hot encoded target
        return (self.y_pred - target_one_hot) * grad_output

    
    def __call__(self, y_pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Allows the CrossEntropyLoss class to be called like a function.
        Computes the Cross-Entropy loss for the provided logits and target labels.

        Args:
            y_pred (np.ndarray): Model predictions (logits).
            target (np.ndarray): Ground truth labels (integer indices).

        Returns:
            float: Computed Cross-Entropy loss value.
        """
        return self.forward(y_pred, target)
