import numpy as np

class SGD:
    """
    Stochastic Gradient Descent with Momentum (SGD) optimizer for training neural networks.

    Stochastic Gradient Descent is an iterative method for optimizing machine learning models.
    The momentum term helps accelerate gradient vectors in the right directions, leading to 
    faster converging during optimization.

    The update rule for each parameter in the model is:
        v(t) = gamma * v(t-1) + learning_rate * grad
        parameter = parameter - v(t)

    where:
        - v(t) is the velocity (momentum term)
        - grad is the gradient of the parameter (weight or bias)
        - gamma is the momentum factor (typically between 0 and 1)
        - learning_rate is the step size for gradient descent

    Example:
        >>> sgd_optimizer = SGD(layers, lr=0.01, gamma=0.9)
        >>> sgd_optimizer.step()
    """
    
    def __init__(self, layers, lr=0.01, gamma=0.9):
        """
        Initializes the Stochastic Gradient Descent optimizer with momentum.

        Args:
            layers (list): List of model layers.
            lr (float, optional): Learning rate for parameter updates. Default is 0.01.
            gamma (float, optional): Momentum factor, controls the influence of previous gradients. Default is 0.9.
        """
        self.layers = layers
        self.lr = lr
        self.gamma = gamma
        self.velocity = {}

    def step(self):
        """
        Performs a single optimization step (parameter update) for each layer.

        For each layer, this method computes the velocity using the momentum term and updates the 
        layer's weights and biases accordingly.

        The update rule for weights and biases:
            - velocity = gamma * previous_velocity + learning_rate * gradient
            - parameter = parameter - velocity

        This method applies the computed gradients stored in each layer and updates the parameters.

        Example:
            >>> sgd_optimizer.step()
        """
        for layer in self.layers:
            if hasattr(layer, "weight") and hasattr(layer, "grad_weight"):
                if id(layer.weight) not in self.velocity:
                    self.velocity[id(layer.weight)] = np.zeros_like(layer.weight)

                if id(layer.bias) not in self.velocity:
                    self.velocity[id(layer.bias)] = np.zeros_like(layer.bias)

                self.velocity[id(layer.weight)] = (
                    self.gamma * self.velocity[id(layer.weight)] + self.lr * layer.grad_weight
                )
                
                self.velocity[id(layer.bias)] = (
                    self.gamma * self.velocity[id(layer.bias)] + self.lr * layer.grad_bias
                )

                layer.weight -= self.velocity[id(layer.weight)]
                layer.bias -= self.velocity[id(layer.bias)].reshape(layer.bias.shape)
