
import triton
import triton.language as tl

@triton.jit
def _slice_4d_dim2_kernel(
    src_ptr, dst_ptr,
    stride_src_0, stride_src_1, stride_src_2, stride_src_3,
    stride_dst_0, stride_dst_1, stride_dst_2, stride_dst_3,
    start_idx,
    D0, D1, D2_dst, D3,
    BLOCK_SIZE: tl.constexpr
):
    # Cast raw integer pointers into proper Triton pointer types.
    src_ptr = src_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    dst_ptr = dst_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    
    # Grid: D0 * D1 * D2_dst
    
    idx_2 = pid % D2_dst
    rem = pid // D2_dst
    idx_1 = rem % D1
    idx_0 = rem // D1
    
    src_idx_2 = idx_2 + start_idx
    
    src_offset = idx_0 * stride_src_0 + idx_1 * stride_src_1 + src_idx_2 * stride_src_2
    dst_offset = idx_0 * stride_dst_0 + idx_1 * stride_dst_1 + idx_2 * stride_dst_2
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D3
    
    # Load and Store
    vals = tl.load(src_ptr + src_offset + offs * stride_src_3, mask=mask)
    tl.store(dst_ptr + dst_offset + offs * stride_dst_3, vals, mask=mask)

def slice_4d_dim2(x, out, start_idx):
    # x: Input Tensor
    # out: Output Tensor (already allocated)
    # start_idx: int
    
    B, H, S_new, D = out.shape
    
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (B * H * S_new,)
    
    _slice_4d_dim2_kernel[grid](
        x.data_ptr, out.data_ptr,
        x.strides[0], x.strides[1], x.strides[2], x.strides[3],
        out.strides[0], out.strides[1], out.strides[2], out.strides[3],
        start_idx,
        B, H, S_new, D,
        BLOCK_SIZE=BLOCK_SIZE
    )
