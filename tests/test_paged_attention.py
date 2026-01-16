import numpy as np
import math
from nano_infer.models.llama import LlamaModel, LlamaConfig
from nano_infer.core.kv_cache import CacheEngine, PagedCacheEngine
from nano_infer.core.tensor import Tensor

def test_paged_attention_vs_static():
    print("Testing Paged Attention vs Static Cache...")
    
    # 1. Setup configuration
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        n_layers=2,
        n_heads=4,
        max_position_embeddings=128,
        max_batch_size=1
    )
    
    device = "cuda"
    
    # Same model weights for both runs
    model = LlamaModel(config)
    model.to(device)
    
    # 2. Static Cache Run
    print("\n--- Running with Static Cache ---")
    static_engine = CacheEngine(config, device=device)
    
    # Prefill (8 tokens)
    input_prefill = np.random.randint(0, config.vocab_size, (1, 8))
    t_prefill = Tensor(input_prefill, device=device)
    model(t_prefill, cache_engine=static_engine)
    
    # Decode (1 token)
    input_decode = np.random.randint(0, config.vocab_size, (1, 1))
    t_decode = Tensor(input_decode, device=device)
    out_static = model(t_decode, cache_engine=static_engine)
    
    out_static_np = out_static.numpy()
    
    # 3. Paged Cache Run
    print("\n--- Running with Paged Cache ---")
    # Block size 4 ensures we span multiple blocks (8 tokens = 2 blocks, +1 token = 3rd block)
    paged_engine = PagedCacheEngine(config, block_size=4, max_num_blocks=100, device=device)
    
    # Reset model? No, model is stateless except for weights.
    # But we need to ensure the sequence of operations is same.
    
    # Prefill (8 tokens)
    # We use the same input tensors
    model(t_prefill, cache_engine=paged_engine)
    
    # Decode (1 token)
    out_paged = model(t_decode, cache_engine=paged_engine)
    
    out_paged_np = out_paged.numpy()
    
    # 4. Compare
    diff = np.abs(out_static_np - out_paged_np).max()
    print(f"\nMax difference between Static and Paged: {diff}")
    
    if diff < 1e-4:
        print("✅ Paged Attention Verified: Matches Static Cache output.")
    else:
        print("❌ Paged Attention Failed: Output mismatch.")
        exit(1)

def test_paged_attention_memory_allocation():
    print("\nTesting Paged Attention Memory Allocation...")
    config = LlamaConfig(hidden_size=64, n_layers=1, n_heads=2)
    engine = PagedCacheEngine(config, block_size=2, max_num_blocks=10)
    
    # Simulating a batch of 1, seq len 5
    # Should allocate ceil(5/2) = 3 blocks
    
    # We need to mock the update call or call it manually via model?
    # Let's call update manually to test logic.
    
    k_dummy = Tensor(np.zeros((1, 5, 2, 32), dtype=np.float32), device="cuda")
    v_dummy = Tensor(np.zeros((1, 5, 2, 32), dtype=np.float32), device="cuda")
    
    engine.update(layer_idx=0, k_new=k_dummy, v_new=v_dummy)
    
    blocks_used = len(engine.block_tables_dict[0])
    print(f"Allocated blocks for seq_len=5, block_size=2: {blocks_used}")
    
    assert blocks_used == 3
    assert engine.context_lens_cpu[0] == 5 # Update now advances context len
    
    engine.advance(5)
    assert engine.context_lens_cpu[0] == 5
    
    # Add 1 more token
    k_inc = Tensor(np.zeros((1, 1, 2, 32), dtype=np.float32), device="cuda")
    v_inc = Tensor(np.zeros((1, 1, 2, 32), dtype=np.float32), device="cuda")
    
    engine.update(layer_idx=0, k_new=k_inc, v_new=v_inc)
    
    blocks_used_new = len(engine.block_tables_dict[0])
    print(f"Allocated blocks after +1 token: {blocks_used_new}")
    
    # 5 tokens used 3 blocks (cap 6). 6th token fits in 3rd block?
    # Block 0: 0,1
    # Block 1: 2,3
    # Block 2: 4, (5 is free)
    # So for 6th token (index 5), it should fit in Block 2.
    # Wait, 5 tokens fill indices 0,1,2,3,4.
    # Block 0: [0, 1]
    # Block 1: [2, 3]
    # Block 2: [4, -]
    # Adding 1 token (index 5) -> fills Block 2.
    # So blocks_used should still be 3.
    
    assert blocks_used_new == 3
    print("✅ Allocation logic verified.")

if __name__ == "__main__":
    test_paged_attention_vs_static()
    test_paged_attention_memory_allocation()
