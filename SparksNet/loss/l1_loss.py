import numpy as np
class L1Loss:
    """
    L1 Loss (Mean Absolute Error) for regression problems. This loss function calculates the average 
    of the absolute differences between the predicted values and the actual target values.
    
    L1 Loss is often used for regression tasks where the model aims to predict continuous values. 
    It is less sensitive to outliers compared to the Mean Squared Error (MSE) and is commonly used 
    in situations where robustness to large errors is desired.

    Definition:
        L1Loss = (1/N) * sum(|y_pred - target|)
    
    where:
        - y_pred = predicted values from the model
        - target = ground truth values
        - N = number of samples
    
    Derivative:
        d(L1Loss)/d(y_pred) = (1/N) * sign(y_pred - target)
    
    The derivative of L1 Loss is used in the backward pass to compute the gradient with respect to 
    the model's predictions, enabling efficient backpropagation.

    Example:
        >>> l1_loss = L1Loss()
        >>> loss = l1_loss.forward(y_pred, target)
        >>> grad = l1_loss.backward(grad_output)
    """
    
    def __init__(self):
        """
        Initializes the L1Loss class.
        Sets up placeholders for model predictions and target values to be used during the 
        backward pass.
        """
        self.y_pred: np.ndarray = None
        self.target: np.ndarray = None

    def forward(self, y_pred: np.ndarray, target: np.ndarray) -> float:
        """
        Forward pass of the L1Loss. Computes the Mean Absolute Error between the predicted 
        and true values.

        Args:
            y_pred (np.ndarray): Model's predicted output.
            target (np.ndarray): Ground truth values.

        Returns:
            float: Computed L1 Loss (Mean Absolute Error).
        """
        self.y_pred = y_pred
        self.target = target
        self.N = self.y_pred.shape[0]
        return np.mean(np.abs(self.y_pred - self.target))

    def backward(self, grad_output: float = 1.0) -> np.ndarray:
        """
        Backward pass of the L1Loss. Computes the gradient of the loss with respect to the 
        predicted values.

        The gradient is computed using the sign of the difference between the predicted values 
        and the target values.

        Args:
            grad_output (float): Gradient from the next layer (default is 1.0).

        Returns:
            np.ndarray: Gradient of the loss with respect to the predicted values.
        """
        grad = np.where(self.y_pred > self.target, 1, -1)
        return (1 / self.N) * grad * grad_output
    
    def __call__(self, y_pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Allows the L1Loss class to be called like a function. This invokes the forward pass 
        to compute the L1 Loss for the provided predicted and target values.

        Args:
            y_pred (np.ndarray): Predicted values from the model.
            target (np.ndarray): Ground truth values.

        Returns:
            float: Computed L1 Loss (Mean Absolute Error).
        """
        return self.forward(y_pred, target)
