import torch
import numpy as np
import nano_infer.ops.functional as F
from nano_infer.core.tensor import Tensor

def test_softmax():
    shape = (1, 4, 32, 32)
    x_np = np.random.randn(*shape).astype(np.float32)
    
    x_pt = torch.tensor(x_np).cuda()
    y_pt = torch.nn.functional.softmax(x_pt, dim=-1)
    
    x_nano = Tensor(x_np).to_cuda()
    y_nano = F.softmax(x_nano, dim=-1)
    
    diff = np.abs(y_pt.cpu().numpy() - y_nano.numpy()).max()
    print(f"Max Difference: {diff}")
    
    if diff < 1e-5:
        print("✅ Softmax Test Passed!")
    else:
        print("❌ Softmax Test Failed!")

if __name__ == "__main__":
    test_softmax()