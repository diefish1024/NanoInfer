import torch
import numpy as np
import nano_infer
from nano_infer.models.llama import LlamaMLP
from nano_infer.core.tensor import Tensor

def test_mlp_correctness():
    B, Seq, H, I = 2, 128, 4096, 11008
    
    x_np = np.random.randn(B, Seq, H).astype(np.float32)

    # ==========================
    # PyTorch baseline
    # ==========================
    x_pt = torch.tensor(x_np).cuda()
    
    class PyTorchMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(H, I, bias=False)
            self.up_proj = torch.nn.Linear(H, I, bias=False)
            self.down_proj = torch.nn.Linear(I, H, bias=False)
        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
            
    model_pt = PyTorchMLP().cuda()
    
    # ==========================
    # NanoInfer
    # ==========================
    model_nano = LlamaMLP(H, I)
    
    model_nano.gate_proj.weight = Tensor(model_pt.gate_proj.weight.detach().cpu().numpy()).to_cuda()
    model_nano.up_proj.weight = Tensor(model_pt.up_proj.weight.detach().cpu().numpy()).to_cuda()
    model_nano.down_proj.weight = Tensor(model_pt.down_proj.weight.detach().cpu().numpy()).to_cuda()
    
    x_nano = Tensor(x_np).to_cuda()
    
    y_pt = model_pt(x_pt).detach().cpu().numpy()
    y_nano = model_nano(x_nano).numpy()
    
    diff = np.abs(y_pt - y_nano).max()
    print(f"Max Difference: {diff}")
    
    if diff < 1e-4:
        print("✅ MLP Test Passed!")
    else:
        print("❌ MLP Test Failed!")

if __name__ == "__main__":
    test_mlp_correctness()