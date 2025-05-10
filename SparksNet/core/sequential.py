class Sequential:
    """
    A container module to stack layers in a linear sequence.

    Each layer is applied in the order they are passed. 
    The output of one layer becomes the input to the next.

    Parameters:
        *layers (Callable): Any number of layer objects that implement
                            a forward `__call__` method and optionally `backward`.

    Example:
        >>> model = Sequential(
        >>>     LinearLayer(2, 4),
        >>>     ReLU(),
        >>>     LinearLayer(4, 1),
        >>>     Sigmoid()
        >>> )
        >>> output = model(X)             # Forward pass
        >>> model.backward(grad_output)   # Backward pass
        >>> params = model.parameters()   # Get trainable layers
    """

    def __init__(self, *layers):
        """
        Initializes the sequence of layers.
        Args:
            *layers: Variable number of layer instances to stack.
        """
        self.layers = layers

    def __call__(self, x):
        """
        Performs the forward pass through all layers.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output after passing through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        """
        Performs backward pass through the network in reverse order.

        Args:
            grad (np.ndarray): Gradient of the loss w.r.t. output.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        """
        Returns the list of trainable layers (those with `.weight`).

        Returns:
            List[object]: Layers containing learnable parameters.
        """
        return [layer for layer in self.layers if hasattr(layer, 'weight')]
