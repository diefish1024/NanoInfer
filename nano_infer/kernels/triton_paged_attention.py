import triton
import triton.language as tl

@triton.jit
def _reshape_and_cache_kernel(
    key_ptr, value_ptr,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    key_stride_b, key_stride_h, key_stride_d,
    value_stride_b, value_stride_h, value_stride_d,
    k_cache_stride_b, k_cache_stride_h, k_cache_stride_s, k_cache_stride_d,
    v_cache_stride_b, v_cache_stride_h, v_cache_stride_s, v_cache_stride_d,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    key_ptr = key_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    value_ptr = value_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    k_cache_ptr = k_cache_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    v_cache_ptr = v_cache_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    slot_mapping_ptr = slot_mapping_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))

    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    block_idx = slot_idx // BLOCK_SIZE
    block_offset = slot_idx % BLOCK_SIZE

    input_offset = token_idx * key_stride_b + head_idx * key_stride_h
    cache_offset = block_idx * k_cache_stride_b + \
                   head_idx * k_cache_stride_h + \
                   block_offset * k_cache_stride_s

    offs_d = tl.arange(0, HEAD_DIM)
    
    k_val = tl.load(key_ptr + input_offset + offs_d * key_stride_d)
    v_val = tl.load(value_ptr + input_offset + offs_d * value_stride_d)
    
    tl.store(k_cache_ptr + cache_offset + offs_d * k_cache_stride_d, k_val)
    tl.store(v_cache_ptr + cache_offset + offs_d * v_cache_stride_d, v_val)

def reshape_and_cache(key, value, k_cache, v_cache, slot_mapping):
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_dim = key.shape[2]
    block_size = k_cache.shape[2]
    
    grid = (num_tokens, num_heads)
    
    _reshape_and_cache_kernel[grid](
        key.data_ptr, value.data_ptr,
        k_cache.data_ptr, v_cache.data_ptr,
        slot_mapping.data_ptr,
        key.strides[0], key.strides[1], key.strides[2],
        value.strides[0], value.strides[1], value.strides[2],
        k_cache.strides[0], k_cache.strides[1], k_cache.strides[2], k_cache.strides[3],
        v_cache.strides[0], v_cache.strides[1], v_cache.strides[2], v_cache.strides[3],
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim
    )

@triton.jit
def _paged_attention_kernel(
    out_ptr,             # [B, H, D]
    exp_sum_ptr,         # [B, H]
    max_logits_ptr,      # [B, H]
    q_ptr,               # [B, H, D]
    k_cache_ptr,         # [NumBlocks, H, BlockSize, D]
    v_cache_ptr,         # [NumBlocks, H, BlockSize, D]
    block_tables_ptr,    # [B, MaxBlocksPerSeq]
    context_lens_ptr,    # [B]
    q_stride_b, q_stride_h, q_stride_d,
    k_cache_stride_b, k_cache_stride_h, k_cache_stride_s, k_cache_stride_d,
    v_cache_stride_b, v_cache_stride_h, v_cache_stride_s, v_cache_stride_d,
    block_tables_stride_b, block_tables_stride_s,
    sm_scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_BLOCKS_PER_SEQ: tl.constexpr
):
    out_ptr = out_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    q_ptr = q_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    k_cache_ptr = k_cache_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    v_cache_ptr = v_cache_ptr.to(tl.int64).to(tl.pointer_type(tl.float32))
    block_tables_ptr = block_tables_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
    context_lens_ptr = context_lens_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))

    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    q_offset = batch_idx * q_stride_b + head_idx * q_stride_h
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + q_offset + offs_d * q_stride_d)
    
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    
    num_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_idx_in_seq in range(0, num_blocks):
        physical_block_idx = tl.load(block_tables_ptr + batch_idx * block_tables_stride_b + block_idx_in_seq * block_tables_stride_s)
        k_block_offset = physical_block_idx * k_cache_stride_b + head_idx * k_cache_stride_h
        
        offs_s = tl.arange(0, BLOCK_SIZE)
        
        k_ptr_base = k_cache_ptr + k_block_offset
        k_ptrs = k_ptr_base + offs_s[:, None] * k_cache_stride_s + offs_d[None, :] * k_cache_stride_d
        k = tl.load(k_ptrs)
        
        scores = tl.sum(q[None, :] * k, axis=1)
        scores *= sm_scale
        
        block_start = block_idx_in_seq * BLOCK_SIZE
        mask = (block_start + offs_s) < context_len
        scores = tl.where(mask, scores, -float('inf'))
        
        block_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_max)
        
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(scores - m_new)
        
        l_new = l_i * alpha + tl.sum(beta, axis=0)
        
        v_ptr_base = v_cache_ptr + (physical_block_idx * v_cache_stride_b + head_idx * v_cache_stride_h)
        v_ptrs = v_ptr_base + offs_s[:, None] * v_cache_stride_s + offs_d[None, :] * v_cache_stride_d
        v = tl.load(v_ptrs)
        
        weighted_v = tl.sum(beta[:, None] * v, axis=0)
        acc = acc * alpha + weighted_v
        
        m_i = m_new
        l_i = l_new
        
    out = acc / l_i
    out_offset = batch_idx * q_stride_b + head_idx * q_stride_h
    tl.store(out_ptr + out_offset + offs_d * q_stride_d, out)

def paged_attention(
    out,
    q,
    k_cache, v_cache,
    block_tables,
    context_lens,
    sm_scale
):
    B, H, D = q.shape
    block_size = k_cache.shape[2]
    max_blocks_per_seq = block_tables.shape[1]
    
    grid = (B, H)
    
    _paged_attention_kernel[grid](
        out.data_ptr,
        None, None,
        q.data_ptr,
        k_cache.data_ptr, v_cache.data_ptr,
        block_tables.data_ptr,
        context_lens.data_ptr,
        q.strides[0], q.strides[1], q.strides[2],
        k_cache.strides[0], k_cache.strides[1], k_cache.strides[2], k_cache.strides[3],
        v_cache.strides[0], v_cache.strides[1], v_cache.strides[2], v_cache.strides[3],
        block_tables.strides[0], block_tables.strides[1],
        sm_scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        MAX_BLOCKS_PER_SEQ=max_blocks_per_seq
    )
