import numpy as np

class DenseLayer:
    def __init__(self, W, b, activation=None):
        self.W = W
        self.b = b
        self.activation = activation

    def forward(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        z = x @ self.W + self.b
        if self.activation is None:
            return z
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'softmax':
            e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return e_z / e_z.sum(axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")