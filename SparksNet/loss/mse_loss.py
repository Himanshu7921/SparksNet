import numpy as np
class MSELoss:
    """
    Mean Squared Error (MSE) Loss for regression tasks. This loss function calculates the 
    average squared difference between the predicted values and the true values. It is widely 
    used for training models in regression tasks, especially when the model's outputs are continuous.

    Definition:
        MSELoss = (1/N) * sum((y_pred - target)^2)
    
    where:
        - y_pred = predicted values from the model
        - target = ground truth values
        - N = number of samples
    
    Derivative:
        d(MSE)/d(y_pred) = (2/N) * (y_pred - target)
    
    The derivative of MSE is used in the backward pass to compute the gradient with respect to 
    the model's predictions for efficient backpropagation.

    Example:
        >>> mse_loss = MSELoss()
        >>> loss = mse_loss.forward(y_pred, target)
        >>> grad = mse_loss.backward(grad_output)
    """
    
    def __init__(self):
        """
        Initializes the MSELoss class.
        Sets up placeholders for model predictions and target values for use during the 
        backward pass.
        """
        self.y_pred: np.ndarray = None
        self.target: np.ndarray = None
        self.N: int = 0

    def forward(self, y_pred: np.ndarray, target: np.ndarray) -> float:
        """
        Forward pass of the MSELoss. Computes the Mean Squared Error between the predicted 
        values and the true values.

        Args:
            y_pred (np.ndarray): Model's predicted output.
            target (np.ndarray): Ground truth values.

        Returns:
            float: Computed MSE Loss.
        """
        self.y_pred = y_pred
        self.target = target
        self.N = self.y_pred.shape[0]
        return np.mean((self.y_pred - self.target) ** 2)

    def backward(self, grad_output: np.ndarray = 1.0) -> np.ndarray:
        """
        Backward pass of the MSELoss. Computes the gradient of the loss with respect to the 
        predicted values.

        The gradient is computed using the derivative of the Mean Squared Error.

        Args:
            grad_output (float or np.ndarray): Gradient from the next layer (default is 1.0).

        Returns:
            np.ndarray: Gradient of the loss with respect to the predicted values.
        """
        grad_input =  (2 / self.N) * (self.y_pred - self.target)
        return grad_input * grad_output
    
    def __call__(self, y_pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Allows the MSELoss class to be called like a function. This invokes the forward pass 
        to compute the MSE Loss for the provided predicted and target values.

        Args:
            y_pred (np.ndarray): Predicted values from the model.
            target (np.ndarray): Ground truth values.

        Returns:
            float: Computed MSE Loss.
        """
        return self.forward(y_pred, target)

    def accuracy(self, y_preds: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the accuracy of the model based on its predictions.

        Args:
            y_preds (np.ndarray): Model's predicted output.
            y_true (np.ndarray): Ground truth values.

        Returns:
            float: Accuracy as a percentage (between 0 and 100).
        """
        return 100 * ((y_preds == y_true).sum().item() / len(y_true))
