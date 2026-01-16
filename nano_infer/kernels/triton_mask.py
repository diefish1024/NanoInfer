import triton
import triton.language as tl

@triton.jit
def _apply_causal_mask_kernel(
    x_ptr,
    stride_row, stride_col,
    Q_Len, K_Len,
    start_pos,
    BLOCK_SIZE: tl.constexpr
):
    # Grid: (B * H * Q_Len) -> flattened rows
    row_idx = tl.program_id(0)
    
    x_ptr = x_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    
    # Determine the query index (relative to current sequence chunk)
    q_idx = row_idx % Q_Len
    
    # Compute the global position of this query token
    global_q_pos = start_pos + q_idx
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    bounds_mask = col_offsets < K_Len
    causal_mask = col_offsets <= global_q_pos
    
    row_start_ptr = x_ptr + row_idx * stride_row
    ptrs = row_start_ptr + col_offsets * stride_col
    
    vals = tl.load(ptrs, mask=bounds_mask, other=float('-inf'))
    vals = tl.where(causal_mask, vals, float('-inf'))
    
    tl.store(ptrs, vals, mask=bounds_mask)

def apply_causal_mask(x, start_pos):
    B, H, Q, K = x.shape
    
    # We flatten B, H, Q into a single "row" dimension
    n_rows = B * H * Q
    
    stride_row = x.strides[-2]
    stride_col = x.strides[-1]
    
    BLOCK_SIZE = triton.next_power_of_2(K)
    
    grid = (n_rows,)
    
    _apply_causal_mask_kernel[grid](
        x.data_ptr,
        stride_row, stride_col,
        Q, K,
        start_pos,
        BLOCK_SIZE=BLOCK_SIZE
    )
