import numpy as np

class Embedding:
    def __init__(self, input_dim, output_dim, initializer="uniform", seed=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = initializer
        self.seed = seed
        self.weights = self.initialize_weights()
    
    def initialize_weights(self):
        rng = np.random.default_rng(self.seed)

        if self.initializer == "uniform":
            limit = 0.05
            return rng.uniform(-limit, limit, size=(self.input_dim, self.output_dim))
        
        elif self.initializer == "normal":
            stddev = 0.05
            return rng.normal(0, stddev, size=(self.input_dim, self.output_dim))
        
        elif self.initializer == "glorot_uniform":
            fan_in = self.input_dim
            fan_out = self.output_dim
            limit = np.sqrt(6 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(self.input_dim, self.output_dim))
        
        elif self.initializer == "glorot_normal":
            fan_in = self.input_dim
            fan_out = self.output_dim
            stddev = np.sqrt(2 / (fan_in + fan_out))
            return rng.normal(0, stddev, size=(self.input_dim, self.output_dim))
        
        else:
            raise ValueError(f"Unknown initializer: {self.initializer}")
    
    def forward(self, indices):
        return self.weights[indices]
    
    def __call__(self, indices):
        return self.forward(indices)
