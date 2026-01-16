import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x,
    N_COLS,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)

    x_ptr = x_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    w_ptr = w_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    out_ptr = out_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))

    row_start_ptr = x_ptr + row_idx * stride_x

    offsets = tl.arange(0, BLOCK_SIZE)
    mask =  offsets < N_COLS

    x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N_COLS
    rsqrt = tl.rsqrt(mean_sq + eps)

    out = x * rsqrt * w

    out_start_ptr = out_ptr + row_idx * stride_x
    tl.store(out_start_ptr + offsets, out, mask=mask)

def rms_norm(x, w, out, eps):
    n_cols = x.shape[-1]
    n_rows = x.numel // n_cols
    row_stride = 1 if len(x.shape) == 1 else x.strides[-2]

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows, )
    
    _rms_norm_kernel[grid](
        x.data_ptr,
        w.data_ptr,
        out.data_ptr,
        row_stride,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out