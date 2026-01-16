import triton
import triton.language as tl

@triton.jit
def _kv_cache_update_kernel(
    k_cache_ptr, v_cache_ptr,
    k_new_ptr, v_new_ptr,
    stride_cb, stride_ch, stride_cs, stride_cd,
    stride_nb, stride_ns, stride_nh, stride_nd,
    start_pos,
    BATCH_SIZE, NEW_SEQ_LEN, NUM_HEADS, HEAD_DIM,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    h_idx = pid % NUM_HEADS
    s_idx = (pid // NUM_HEADS) % NEW_SEQ_LEN
    b_idx = pid // (NUM_HEADS * NEW_SEQ_LEN)

    k_cache_ptr = k_cache_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    v_cache_ptr = v_cache_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    k_new_ptr = k_new_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    v_new_ptr = v_new_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))

    # Compute source memory offset for [B, S_new, H, D] layout.
    src_offset = b_idx * stride_nb + s_idx * stride_ns + h_idx * stride_nh
    
    # Compute destination memory offset for [B, H, Max_S, D] layout at start_pos.
    dst_offset = b_idx * stride_cb + h_idx * stride_ch + (start_pos + s_idx) * stride_cs

    offs_d = tl.arange(0, BLOCK_SIZE)
    mask = offs_d < HEAD_DIM

    k_vals = tl.load(k_new_ptr + src_offset + offs_d * stride_nd, mask=mask)
    tl.store(k_cache_ptr + dst_offset + offs_d * stride_cd, k_vals, mask=mask)
    
    v_vals = tl.load(v_new_ptr + src_offset + offs_d * stride_nd, mask=mask)
    tl.store(v_cache_ptr + dst_offset + offs_d * stride_cd, v_vals, mask=mask)

def kv_cache_update(k_cache, v_cache, k_new, v_new, start_pos):
    B, S_new, H, D = k_new.shape
    
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (B * S_new * H,)

    _kv_cache_update_kernel[grid](
        k_cache.data_ptr, v_cache.data_ptr,
        k_new.data_ptr, v_new.data_ptr,
        *k_cache.strides,
        *k_new.strides,
        start_pos,
        B, S_new, H, D,
        BLOCK_SIZE=BLOCK_SIZE
    )