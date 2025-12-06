import torch
import numpy as np
import nano_infer
from nano_infer.models.llama import LlamaMLP
from nano_infer.core.tensor import Tensor

def test_mlp_correctness():
    # 1. 设定参数
    B, Seq, H, I = 2, 128, 4096, 11008
    
    # 2. 准备随机输
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
    
    # 把 PyTorch 的权重复制给 NanoInfer，确保起点一致
    model_nano.gate_proj.weight.data = Tensor(model_pt.gate_proj.weight.detach().cpu().numpy()).to_cuda().data
    model_nano.up_proj.weight.data = Tensor(model_pt.up_proj.weight.detach().cpu().numpy()).to_cuda().data
    model_nano.down_proj.weight.data = Tensor(model_pt.down_proj.weight.detach().cpu().numpy()).to_cuda().data
    
    x_nano = Tensor(x_np).to_cuda()
    
    # 3. 运行前向传播
    y_pt = model_pt(x_pt).detach().cpu().numpy()
    y_nano = model_nano(x_nano).numpy()
    
    # 4. 验证误差
    diff = np.abs(y_pt - y_nano).max()
    print(f"Max Difference: {diff}")
    
    if diff < 1e-4:
        print("✅ MLP Test Passed!")
    else:
        print("❌ MLP Test Failed!")

if __name__ == "__main__":
    test_mlp_correctness()