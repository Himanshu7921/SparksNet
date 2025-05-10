import numpy as np

class RMSprop:
    """
    RMSprop Optimizer for training neural networks.

    RMSprop (Root Mean Square Propagation) adapts the learning rate for each parameter by dividing 
    the gradient by a running average of its recent magnitudes.

    Update rule per parameter:
        v(t) = β * v(t-1) + (1 - β) * (grad ** 2)
        param -= lr * grad / (sqrt(v(t) + ε))

    Example:
        >>> model = [Linear(input_dim=128, output_dim=10), ReLU()]
        >>> optimizer = RMSprop(model, lr=0.001)
        >>> optimizer.step()
    """
    
    def __init__(self, layers: list, lr=0.001, beta=0.9, epsilon=1e-8):
        self.layers = layers
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon

        self.v = {}  # Running average of squared gradients
        self.t = 0   # Time step

    def step(self):
        """Performs a single optimization step on all layers."""
        self.t += 1

        for layer in self.layers:
            for param_name in ['weight', 'bias']:
                if hasattr(layer, param_name) and hasattr(layer, f'grad_{param_name}'):
                    param = getattr(layer, param_name)
                    grad = getattr(layer, f'grad_{param_name}')
                    param_id = id(param)

                    # Safety check: shapes must match
                    assert param.shape == grad.shape, (
                        f"Shape mismatch: param {param_name} shape {param.shape}, "
                        f"grad shape {grad.shape}"
                    )

                    # Initialize moments if first time
                    if param_id not in self.v:
                        self.v[param_id] = np.zeros_like(param)

                    # Update running average of squared gradients
                    self.v[param_id] = self.beta * self.v[param_id] + (1 - self.beta) * (grad ** 2)

                    # Parameter update
                    update = self.lr * grad / (np.sqrt(self.v[param_id]) + self.epsilon)
                    getattr(layer, param_name)[...] -= update
