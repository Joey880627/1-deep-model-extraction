import numpy as np
from utils import *

class Model:
    
    def __init__(self, input_dim=2, hidden_dim=2, weights=None, bias=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if weights and bias:
            self.weights = weights
            self.bias = bias
        else:
            self.weights = [np.random.randn(self.input_dim, self.hidden_dim), np.random.randn(self.hidden_dim)]
            self.bias = [np.random.randn(self.hidden_dim), np.random.randn(1)]
        
    def forward(self, x):
        assert x.shape == (self.input_dim,), f"Expected data dimension: ({self.input_dim},), found {x.shape}"
        # Input -> Hidden
        hidden_representation = relu(self.weights[0].T @ x + self.bias[0])
        # Hidden -> Output
        output = self.weights[1].T @ hidden_representation + self.bias[1]
        return output.item()
    
    def __call__(self, x):
        return self.forward(x)
    
    def __str__(self):
        ret = ""
        ret += "================ Model Parameters ================\n"
        ret += f"Input dim: {self.input_dim}\t"
        ret += f"Hidden dim: {self.hidden_dim}\t"
        ret += f"Output dim: 1\n"
        ret += "==================================================\n"
        ret += f"First layer weights:\n\n{self.weights[0]}\n\n"
        ret += f"First layer bias:\n\n{self.bias[0]}\n\n"
        ret += f"Output layer weights:\n\n{self.weights[1]}\n\n"
        ret += f"Output layer bias:\n\n{self.bias[1]}\n"
        ret += "==================================================\n"
        return ret