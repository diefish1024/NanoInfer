import triton
import triton.language as tl

@triton.jit
def _embedding_kernel(
    idx_ptr,
    w_ptr,
    out_ptr,
    N_IDX,
    stride_w_row,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    if pid >= N_IDX:
        return
    
    idx_ptr = idx_ptr.to(tl.pointer_type(tl.int32))
    w_ptr = w_ptr.to(tl.pointer_type(tl.float32))
    out_ptr = out_ptr.to(tl.pointer_type(tl.float32))

    idx = tl.load(idx_ptr + pid)

    src_row_st = w_ptr + idx * stride_w_row 
    dst_row_st = out_ptr + pid * HIDDEN_DIM

    for off in range(0, HIDDEN_DIM, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < HIDDEN_DIM

        val = tl.load(src_row_st + cols, mask=mask)
        tl.store(dst_row_st + cols, val, mask=mask)

def embedding(idx, w, out):
    n_idx = idx.numel
    hidden_dim = w.shape[1]

    gride = (n_idx, )
    BLOCK_SIZE = 1024
    _embedding_kernel[gride](
        idx.data_ptr,
        w.data_ptr,
        out.data_ptr,
        n_idx,
        w.strides[0],
        HIDDEN_DIM=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
