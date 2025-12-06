import nano_infer.nn as nn
from nano_infer.ops import functional as F

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.silu(gate)
        up = self.up_proj(x)
        fuse = gate * up
        out = self.down_proj(fuse)
        return out