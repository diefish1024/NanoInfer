from .modules import Module
from .parameter import Parameter
from ..ops import functional as F
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        k = np.sqrt(1.0 / in_features)
        self.weight = Parameter(np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32))

        if bias:
            self.bias = Parameter(np.random.uniform(-k, k, (out_features,)).astype(np.float32))
        else:
            self.bias = None

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)