from .modules import Module
from .parameter import Parameter
from ..ops import functional as F
import numpy as np

class RMSNorm(Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = Parameter(np.ones(hidden_size, dtype=np.float32))

    def forward(self, x):
        return F.rms_norm(x, self.weight, self.eps)