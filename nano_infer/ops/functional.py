
import numpy as np
from typing import Optional
from ..core.tensor import Tensor
from ..kernels import triton_norm, triton_math, triton_embedding, triton_softmax, triton_rope, triton_kvcache, triton_slice, triton_permute, triton_paged_attention, triton_mask
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
    if x.device != weight.device:
        raise RuntimeError(f"Device mismatch: Input is on {x.device} but weight is on {weight.device}. ")
    if x.device != Device.CUDA:
        raise NotImplementedError("CPU implementation (AVX) is coming soon.")

    out = Tensor.empty(x.shape, x.dtype, device=x.device)

    triton_norm.rms_norm(x, weight, out, eps)

    return out

def silu(x: Tensor) -> Tensor:
    out = Tensor.empty(x.shape, x.dtype, device=x.device)
    if x.device == Device.CPU:
        CppTensor.silu(x.data, out.data);
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
    vocab_size, hidden_dim = weight.shape
    out_shape = list(input.shape) + [hidden_dim]
    out = Tensor.empty(out_shape, weight.dtype, device=input.device)

    if input.device == Device.CUDA:
        triton_embedding.embedding(input, weight, out)
    elif input.device == Device.CPU:
        CppTensor.embedding(input.data, weight.data, out.data)

    return out

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    if dim != -1 and dim != len(x.shape) - 1:
        raise NotImplementedError("Softmax currently only supports dim=-1")
    out = Tensor.empty(x.shape, x.dtype, device=x.device)
    
    if x.device == Device.CUDA:
        triton_softmax.softmax(x, out)
    elif x.device == Device.CPU:
        CppTensor.softmax(x, out)
    
    return out

def rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, start_pos: int = 0) -> tuple[Tensor, Tensor]:
    # in-place
    if q.device == Device.CUDA:
        triton_rope.rope(q, k, cos, sin, start_pos)
    elif q.device == Device.CPU:
        CppTensor.rope(q.data, k.data, cos.data, sin.data, start_pos)
        
    return q, k

def kv_cache_update(k_cache: Tensor, v_cache: Tensor, k_src: Tensor, v_src: Tensor, start_pos: int):
    if k_cache.device == Device.CUDA:
        triton_kvcache.kv_cache_update(k_cache, v_cache, k_src, v_src, start_pos)
    elif k_cache.device == Device.CPU:
        CppTensor.kv_cache_update(k_cache.data, v_cache.data, k_src.data, v_src.data, start_pos)

def reshape_and_cache(key: Tensor, value: Tensor, k_cache: Tensor, v_cache: Tensor, slot_mapping: Tensor):
    if key.device == Device.CUDA:
        triton_paged_attention.reshape_and_cache(key, value, k_cache, v_cache, slot_mapping)
    else:
        raise NotImplementedError("CPU reshape_and_cache not implemented")

def paged_attention(out: Tensor, q: Tensor, k_cache: Tensor, v_cache: Tensor, block_tables: Tensor, context_lens: Tensor, sm_scale: float):
    if q.device == Device.CUDA:
        triton_paged_attention.paged_attention(out, q, k_cache, v_cache, block_tables, context_lens, sm_scale)

def slice(x: Tensor, dim: int, start: int, end: int) -> Tensor:
    if len(x.shape) == 4 and dim == 2:
        B, H, S, D = x.shape
        length = end - start
        out = Tensor.empty((B, H, length, D), x.dtype, device=x.device)
        triton_slice.slice_4d_dim2(x, out, start)
        return out
    else:
        raise NotImplementedError("Only 4D dim=2 slice is implemented")

def transpose(x: Tensor, dim0: int, dim1: int) -> Tensor:
    ndim = len(x.shape)
    if dim0 < 0: dim0 += ndim
    if dim1 < 0: dim1 += ndim
    
    if x.device == Device.CUDA:
        if len(x.shape) == 4:
            if (dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1):
                B, D1, D2, D3 = x.shape
                out = Tensor.empty((B, D2, D1, D3), x.dtype, device=x.device)
                triton_permute.permute_0213(x, out)
                return out
            elif (dim0 == 2 and dim1 == 3) or (dim0 == 3 and dim1 == 2):
                B, H, S, D = x.shape
                out = Tensor.empty((B, H, D, S), x.dtype, device=x.device)
                triton_permute.permute_0132(x, out)
                return out
            else:
                raise NotImplementedError(f"Transpose for shape {x.shape} dims {dim0},{dim1} not implemented yet")
        elif len(x.shape) == 3:
            # Handle 3D transpose by viewing as 4D with B=1
            # Map dims: 0->1, 1->2, 2->3
            d0_4d = dim0 + 1
            d1_4d = dim1 + 1
            
            if (d0_4d == 2 and d1_4d == 3) or (d0_4d == 3 and d1_4d == 2):
                # Swap last two dims (1 and 2 in 3D -> 2 and 3 in 4D)
                # Input: [D0, D1, D2] -> View [1, D0, D1, D2]
                # Output: [1, D0, D2, D1] -> View [D0, D2, D1]
                # Corresponds to permute_0132 on 4D
                D0, D1, D2 = x.shape
                x_view = x.view(1, D0, D1, D2)
                out = Tensor.empty((1, D0, D2, D1), x.dtype, device=x.device)
                triton_permute.permute_0132(x_view, out)
                return out.view(D0, D2, D1)
            
            elif (d0_4d == 1 and d1_4d == 2) or (d0_4d == 2 and d1_4d == 1):
                # Swap first two dims (0 and 1 in 3D -> 1 and 2 in 4D)
                # Input: [D0, D1, D2] -> View [1, D0, D1, D2]
                # Output: [1, D1, D0, D2] -> View [D1, D0, D2]
                # Corresponds to permute_0213 on 4D
                D0, D1, D2 = x.shape
                x_view = x.view(1, D0, D1, D2)
                out = Tensor.empty((1, D1, D0, D2), x.dtype, device=x.device)
                triton_permute.permute_0213(x_view, out)
                return out.view(D1, D0, D2)
            else:
                 raise NotImplementedError(f"Transpose for 3D shape {x.shape} dims {dim0},{dim1} not implemented yet")
        else:
            raise NotImplementedError(f"Transpose for shape {x.shape} dims {dim0},{dim1} not implemented yet")
    else:
        raise NotImplementedError("CPU transpose not implemented")

def apply_causal_mask(x: Tensor, start_pos: int):
    # In-place modification of x (scores)
    if x.device == Device.CUDA:
        triton_mask.apply_causal_mask(x, start_pos)
    else:
        raise NotImplementedError("CPU apply_causal_mask not implemented")
