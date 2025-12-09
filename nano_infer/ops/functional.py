import numpy as np
from typing import Optional
from ..core.tensor import Tensor
from ..kernels import triton_norm, triton_math, triton_embedding
from ..backend import _nano_infer

CppTensor = _nano_infer.Tensor
Device = _nano_infer.Device

def add(x: Tensor, other: Tensor) -> Tensor:
    out_cpp = x.data.add(other.data)
    return Tensor(out_cpp, device=Device.CUDA)

def mul(x: Tensor, other: Tensor) -> Tensor:
    out_cpp = x.data.mul(other.data)
    return Tensor(out_cpp, device=Device.CUDA)

def matmul(x: Tensor, other: Tensor, trans_a=False, trans_b=False) -> Tensor:
    out_cpp = x.data.matmul(other.data, trans_a, trans_b)
    return Tensor(out_cpp, device=Device.CUDA)

def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    if x.device != Device.CUDA:
        raise NotImplementedError("CPU implementation (AVX) is coming soon in Milestone 2.")

    # TODO: implement empty_like in C++ Tensor
    out = Tensor(np.empty(x.shape, dtype=np.float32), device=Device.CUDA)

    triton_norm.rms_norm(x, weight, out, eps)

    return out

def silu(x: Tensor) -> Tensor:    
    out = Tensor(np.empty(x.shape, dtype=np.float32), device=Device.CUDA)
    if x.device == Device.CPU:
        CppTensor.silu(x, out);
    elif x.device == Device.CUDA:
        triton_math.silu(x, out)
    else:
        raise RuntimeError(f"SiLU not implemented for device: {x.device}")
    return out

def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None):
    # input @ weight^T + bias
    output = matmul(input, weight, trans_a=False, trans_b=True)
    
    if bias is not None:
        output = output + bias
        
    return output

def embedding(input: Tensor, weight: Tensor) -> Tensor:
    batch_size, seq_len = input.shape
    vocab_size, hidden_dim = weight.shape
    out_shape = list(input.shape) + [hidden_dim]
    out = Tensor(np.empty(out_shape, dtype=np.float32), device=input.device)

    if input.device == Device.CUDA:
        triton_embedding.embedding(input, weight, out)
    elif input.device == Device.CPU:
        CppTensor.embedding(input.data, weight.data, out.data)

    return out