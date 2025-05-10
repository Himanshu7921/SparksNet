import numpy as np

class Adam:
    """
    Adam Optimizer for training neural networks.

    Adam (Adaptive Moment Estimation) combines the benefits of Momentum and RMSProp. It maintains 
    exponentially decaying averages of past gradients (first moment) and squared gradients 
    (second moment) to adaptively update each parameter.

    Update rule per parameter:
        m(t) = β1 * m(t-1) + (1 - β1) * grad
        v(t) = β2 * v(t-1) + (1 - β2) * (grad ** 2)
        m̂ = m(t) / (1 - β1 ** t)
        v̂ = v(t) / (1 - β2 ** t)
        param -= lr * m̂ / (sqrt(v̂) + ε)

    Example:
        >>> model = [Linear(input_dim=128, output_dim=10), ReLU()]
        >>> optimizer = Adam(model, lr=0.001)
        >>> optimizer.step()
    """

    def __init__(self, layers: list, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-9):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}  # First moment
        self.v = {}  # Second moment
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
                    if param_id not in self.m:
                        self.m[param_id] = np.zeros_like(param)
                        self.v[param_id] = np.zeros_like(param)

                    # Update moments
                    self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
                    self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)

                    # Bias correction
                    m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

                    # Parameter update
                    update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    getattr(layer, param_name)[...] -= update

