import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    stride_input_row, stride_output_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    input_ptr = input_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    
    row_start_ptr = input_ptr + row_idx * stride_input_row
    out_row_start_ptr = output_ptr + row_idx * stride_output_row
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=mask)


def softmax(x, out):
    n_rows = x.numel // x.shape[-1]
    n_cols = x.shape[-1]
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    grid = (n_rows, )
    stride_in = x.strides[-2] if len(x.shape) > 1 else 1
    stride_out = out.strides[-2] if len(out.shape) > 1 else 1

    _softmax_kernel[grid](
        x.data_ptr,
        out.data_ptr,
        stride_in, stride_out,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )