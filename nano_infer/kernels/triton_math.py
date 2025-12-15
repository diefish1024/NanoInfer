import triton
import triton.language as tl

@triton.jit
def _silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x_ptr = x_ptr.to(tl.int64)
    out_ptr = out_ptr.to(tl.int64)

    x_ptr = x_ptr.to(tl.pointer_type(tl.float32))
    out_ptr = out_ptr.to(tl.pointer_type(tl.float32))
    
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    output = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, output, mask=mask)

def silu(x, out):
    n_elements = x.numel
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    _silu_kernel[grid](
        x.data_ptr,
        out.data_ptr,
        n_elements,
        BLOCK_SIZE=1024
    )
    return out