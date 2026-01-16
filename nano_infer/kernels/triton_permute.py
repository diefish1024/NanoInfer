
import triton
import triton.language as tl

@triton.jit
def _permute_0213_kernel(
    src_ptr, dst_ptr,
    stride_src_0, stride_src_1, stride_src_2, stride_src_3,
    stride_dst_0, stride_dst_1, stride_dst_2, stride_dst_3,
    D0, D1, D2, D3,
    BLOCK_SIZE: tl.constexpr
):
    src_ptr = src_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))

    # Transpose dim 1 and 2: [D0, D1, D2, D3] -> [D0, D2, D1, D3]
    pid = tl.program_id(0)
    
    idx_1 = pid % D1
    rem = pid // D1
    idx_2 = rem % D2
    idx_0 = rem // D2
    
    src_offset = idx_0 * stride_src_0 + idx_1 * stride_src_1 + idx_2 * stride_src_2
    dst_offset = idx_0 * stride_dst_0 + idx_2 * stride_dst_1 + idx_1 * stride_dst_2
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D3
    
    vals = tl.load(src_ptr + src_offset + offs * stride_src_3, mask=mask)
    tl.store(dst_ptr + dst_offset + offs * stride_dst_3, vals, mask=mask)

def permute_0213(x, out):
    B, S, H, D = x.shape
    
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (B * H * S,)
    
    _permute_0213_kernel[grid](
        x.data_ptr, out.data_ptr,
        x.strides[0], x.strides[1], x.strides[2], x.strides[3],
        out.strides[0], out.strides[1], out.strides[2], out.strides[3],
        B, S, H, D,
        BLOCK_SIZE=BLOCK_SIZE
    )

@triton.jit
def _permute_0132_kernel(
    src_ptr, dst_ptr,
    stride_src_0, stride_src_1, stride_src_2, stride_src_3,
    stride_dst_0, stride_dst_1, stride_dst_2, stride_dst_3,
    D0, D1, D2, D3,
    BLOCK_SIZE: tl.constexpr
):
    src_ptr = src_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))

    # Transpose dim 2 and 3: [D0, D1, D2, D3] -> [D0, D1, D3, D2]
    
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    idx_3 = pid_row % D3
    rem = pid_row // D3
    idx_1 = rem % D1
    idx_0 = rem // D1
    
    offs_2 = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_2 < D2
    
    src_offset = idx_0 * stride_src_0 + idx_1 * stride_src_1 + offs_2 * stride_src_2 + idx_3 * stride_src_3
    dst_offset = idx_0 * stride_dst_0 + idx_1 * stride_dst_1 + idx_3 * stride_dst_2 + offs_2 * stride_dst_3
    
    vals = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, vals, mask=mask)

def permute_0132(x, out):
    B, H, S, D = x.shape
    
    BLOCK_SIZE = triton.next_power_of_2(S)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    if BLOCK_SIZE < 32:
        BLOCK_SIZE = 32
        
    grid = (B * H * D, triton.cdiv(S, BLOCK_SIZE))
    
    _permute_0132_kernel[grid](
        x.data_ptr, out.data_ptr,
        x.strides[0], x.strides[1], x.strides[2], x.strides[3],
        out.strides[0], out.strides[1], out.strides[2], out.strides[3],
        B, H, S, D,
        BLOCK_SIZE=BLOCK_SIZE
    )
