import triton
import triton.language as tl

@triton.jit
def _rope_kernel(
    q_ptr, k_ptr,
    cos_ptr, sin_ptr,
    
    # Strides
    stride_q_batch, stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_seq, stride_k_head, stride_k_dim,
    stride_cos_seq, stride_cos_dim,
    
    # Params
    start_pos,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    pid_token = tl.program_id(0)
    pid_head  = tl.program_id(1)
    
    batch_idx = pid_token // SEQ_LEN
    seq_idx   = pid_token % SEQ_LEN
    
    q_offset = batch_idx * stride_q_batch + seq_idx * stride_q_seq + pid_head * stride_q_head
    k_offset = batch_idx * stride_k_batch + seq_idx * stride_k_seq + pid_head * stride_k_head
    
    global_pos = start_pos + seq_idx
    rope_offset = global_pos * stride_cos_seq
    
    HALF_DIM = HEAD_DIM // 2
    off_half = tl.arange(0, BLOCK_SIZE)
    mask = off_half < HALF_DIM

    q_ptr = q_ptr.to(tl.pointer_type(tl.float32))
    k_ptr = k_ptr.to(tl.pointer_type(tl.float32))
    cos_ptr = cos_ptr.to(tl.pointer_type(tl.float32))
    sin_ptr = sin_ptr.to(tl.pointer_type(tl.float32))
    
    c = tl.load(cos_ptr + rope_offset + off_half, mask=mask, other=0.0)
    s = tl.load(sin_ptr + rope_offset + off_half, mask=mask, other=0.0)
    
    # assume stride Q = 1 (to be changed)
    q1_ptr = q_ptr + q_offset + off_half
    q2_ptr = q_ptr + q_offset + HALF_DIM + off_half
    
    q1 = tl.load(q1_ptr, mask=mask, other=0.0)
    q2 = tl.load(q2_ptr, mask=mask, other=0.0)

    k1_ptr = k_ptr + k_offset + off_half
    k2_ptr = k_ptr + k_offset + HALF_DIM + off_half
    
    k1 = tl.load(k1_ptr, mask=mask, other=0.0)
    k2 = tl.load(k2_ptr, mask=mask, other=0.0)
    
    # out1 = x1 * cos - x2 * sin
    # out2 = x2 * cos + x1 * sin
    q1_out = q1 * c - q2 * s
    q2_out = q2 * c + q1 * s
    
    k1_out = k1 * c - k2 * s
    k2_out = k2 * c + k1 * s
    
    tl.store(q1_ptr, q1_out, mask=mask)
    tl.store(q2_ptr, q2_out, mask=mask)
    
    tl.store(k1_ptr, k1_out, mask=mask)
    tl.store(k2_ptr, k2_out, mask=mask)

def rope(q, k, cos, sin, start_pos=0):
    # q, k: [Batch, Seq, Heads, Dim]
    BATCH, SEQ, HEADS, DIM = q.shape
    HALF_DIM = DIM // 2
    
    BLOCK_SIZE = triton.next_power_of_2(HALF_DIM)
    
    grid = (BATCH * SEQ, HEADS)
    
    _rope_kernel[grid](
        q.data_ptr, k.data_ptr,
        cos.data_ptr, sin.data_ptr,
        
        # Strides
        q.strides[0], q.strides[1], q.strides[2], q.strides[3],
        k.strides[0], k.strides[1], k.strides[2], k.strides[3],
        cos.strides[0], cos.strides[1],
        
        # Scalar Params
        start_pos,
        SEQ,
        DIM,
        BLOCK_SIZE=BLOCK_SIZE
    )