import numpy as np

class SparksTensor:
    """
    A lightweight tensor class that mimics behavior of PyTorch tensors,
    internally backed by NumPy arrays.
    """

    __module__ = 'SparksNet'
    __qualname__ = 'Tensor'

    def __init__(self, data):
        """
        Initialize the tensor with given data.

        Args:
            data (list | int | float | np.ndarray): Input data to convert into tensor.
        """
        self.data = np.array(data)

    @property
    def shape(self):
        """
        Returns:
            tuple: Shape of the underlying NumPy array.
        """
        return self.data.shape

    @property
    def dtype(self):
        """
        Returns:
            np.dtype: Data type of the underlying NumPy array.
        """
        return self.data.dtype

    def __add__(self, other):
        """
        Add two tensors or tensor with scalar.

        Args:
            other (SparksTensor | int | float): Right-hand operand.

        Returns:
            SparksTensor: Result of element-wise addition.
        """
        if isinstance(other, SparksTensor):
            result = self.data + other.data
        else:
            result = self.data + other

        return SparksTensor(result)

    def __radd__(self, other):
        """
        Right-side addition support.

        Args:
            other (int | float): Scalar value.

        Returns:
            SparksTensor: Result of scalar + tensor.
        """
        return self + other

    def __sub__(self, other):
        """
        Subtract two tensors or tensor with scalar.

        Args:
            other (SparksTensor | int | float): Right-hand operand.

        Returns:
            SparksTensor: Result of element-wise subtraction.
        """
        if isinstance(other, SparksTensor):
            result = self.data - other.data
        else:
            result = self.data - other

        return SparksTensor(result)

    def __rsub__(self, other):
        """
        Right-side subtraction support (scalar - tensor).

        Args:
            other (int | float): Scalar value.

        Returns:
            SparksTensor: Result of scalar - tensor.
        """
        return SparksTensor(other - self.data)

    def __mul__(self, other):
        """
        Multiply two tensors or tensor with scalar.

        Args:
            other (SparksTensor | int | float): Right-hand operand.

        Returns:
            SparksTensor: Result of element-wise multiplication.
        """
        if isinstance(other, SparksTensor):
            result = self.data * other.data
        else:
            result = self.data * other

        return SparksTensor(result)

    def __rmul__(self, other):
        """
        Right-side multiplication support.

        Args:
            other (int | float): Scalar value.

        Returns:
            SparksTensor: Result of scalar * tensor.
        """
        return self * other

    def __truediv__(self, other):
        """
        Element-wise true division.

        Args:
            other (SparksTensor | int | float): Right-hand operand.

        Returns:
            SparksTensor: Result of division.

        Raises:
            ZeroDivisionError: If dividing by zero.
        """
        if isinstance(other, SparksTensor):
            if np.any(other.data == 0):
                raise ZeroDivisionError("division by Zero")
            result = self.data / other.data
        else:
            if other == 0:
                raise ZeroDivisionError("division by Zero")
            result = self.data / other
        return SparksTensor(result)

    def __rtruediv__(self, other):
        """
        Right-side true division support (scalar / tensor).

        Args:
            other (int | float): Scalar numerator.

        Returns:
            SparksTensor: Result of scalar / tensor.

        Raises:
            ZeroDivisionError: If any tensor element is zero.
        """
        if np.any(self.data == 0):
            raise ZeroDivisionError("division by Zero")
        return SparksTensor(other / self.data)

    def numpy(self):
        """
        Returns:
            np.ndarray: Underlying NumPy array.
        """
        return self.data

    def __repr__(self):
        """
        Returns:
            str: A clean printable string of the tensor.
        """
        return f"net.Tensor({self.data})"
    
    @staticmethod
    def zeros(shape):
        """
        Create a tensor filled with zeros.
        
        Args:
            shape (tuple): Shape of the tensor.
            
        Returns:
            SparksTensor: Tensor with all zeros.
        """
        return SparksTensor(np.zeros(shape))
    
    @staticmethod
    def ones(shape):
        """
        Create a tensor filled with ones.
        
        Args:
            shape (tuple): Shape of the tensor.
            
        Returns:
            SparksTensor: Tensor with all ones.
        """
        return SparksTensor(np.ones(shape))
    
    @staticmethod
    def rand(shape):
        """
        Create a tensor filled with random numbers from 0 to 1.
        
        Args:
            shape (tuple): Shape of the tensor.
            
        Returns:
            SparksTensor: Tensor with all random numbers.
        """
        return SparksTensor(np.random.rand(*shape))
    
    @staticmethod
    def eye(shape):
        """
        Create an identity tensor.

        Args:
            shape (int or tuple): Shape of the identity tensor. If tuple, must be (rows, cols).
                
        Returns:
            SparksTensor: Identity matrix.
        """
        if isinstance(shape, int):
            return SparksTensor(np.eye(shape))
        
        if isinstance(shape, tuple) and len(shape) == 2:
            return SparksTensor(np.eye(*shape))
        else:
            raise ValueError("eye() expects an int or a tuple of two ints")

