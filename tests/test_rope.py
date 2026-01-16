import numpy as np
from nano_infer.core.tensor import Tensor
import nano_infer.ops.functional as F

def test_rope():
    B, S, H, D = 1, 4, 2, 64
    start_pos = 10
    
    q_np = np.random.randn(B, S, H, D).astype(np.float32)
    k_np = np.random.randn(B, S, H, D).astype(np.float32)
    
    max_seq = 100
    freqs = np.random.randn(max_seq, D // 2).astype(np.float32)
    cos_np = np.cos(freqs).astype(np.float32)
    sin_np = np.sin(freqs).astype(np.float32)
    
    q_nano = Tensor(q_np.copy()).to_cuda()
    k_nano = Tensor(k_np.copy()).to_cuda()
    cos_nano = Tensor(cos_np).to_cuda()
    sin_nano = Tensor(sin_np).to_cuda()
    
    F.rope(q_nano, k_nano, cos_nano, sin_nano, start_pos)
    
    # Llama Rotate Half:
    # x1 = x[..., :D/2], x2 = x[..., D/2:]
    # out = [x1*c - x2*s, x2*c + x1*s]
    
    cur_cos = cos_np[start_pos : start_pos + S, :] # [S, D/2]
    cur_sin = sin_np[start_pos : start_pos + S, :]
    
    # Heads [1, S, 1, D/2]
    cur_cos = cur_cos[None, :, None, :]
    cur_sin = cur_sin[None, :, None, :]
    
    q_half_1 = q_np[..., :D//2]
    q_half_2 = q_np[..., D//2:]
    
    q_out_1 = q_half_1 * cur_cos - q_half_2 * cur_sin
    q_out_2 = q_half_2 * cur_cos + q_half_1 * cur_sin
    q_gt = np.concatenate([q_out_1, q_out_2], axis=-1)
    
    diff = np.abs(q_nano.numpy() - q_gt).max()
    print(f"Max Difference: {diff}")
    
    if diff < 1e-5:
        print("✅ RoPE Test Passed")
    else:
        print("❌ RoPE Test Failed")

if __name__ == "__main__":
    test_rope()