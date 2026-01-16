import numpy as np
import math
from nano_infer.models.llama import LlamaModel, LlamaConfig
from nano_infer.core.kv_cache import CacheEngine
from nano_infer.core.tensor import Tensor

def test_kv_cache_logic():
    # 1. Setup minimal configuration for testing.
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        n_layers=2,
        n_heads=4,
        max_position_embeddings=128,
        max_batch_size=1
    )
    
    device = "cuda" # or "cpu"
    model = LlamaModel(config)
    model.to(device)
    engine = CacheEngine(config, device=device)
    
    # 2. Prefill Phase: Input a sequence of 8 tokens.
    print("Running Prefill...")
    input_prefill = np.random.randint(0, config.vocab_size, (1, 8))
    t_prefill = Tensor(input_prefill, device=device)
    
    # Forward pass updates the cache and advances current_pos to 8.
    output_prefill = model(t_prefill, cache_engine=engine)
    
    assert engine.current_pos == 8
    print(f"Prefill successful. Cache position: {engine.current_pos}")

    # 3. Decode Phase: Input 1 new token.
    print("\nRunning Decode Step 1...")
    input_decode = np.random.randint(0, config.vocab_size, (1, 1))
    t_decode = Tensor(input_decode, device=device)
    
    # Forward pass updates the cache and advances current_pos to 9.
    output_decode = model(t_decode, cache_engine=engine)
    
    assert engine.current_pos == 9
    print(f"Decode Step 1 successful. Cache position: {engine.current_pos}")

    # 4. Consistency Check: Full sequence vs. Incremental.
    print("\nVerifying Consistency...")
    # Concatenate original prefill and the new token.
    full_seq = np.concatenate([input_prefill, input_decode], axis=1)
    t_full = Tensor(full_seq, device=device)
    
    # Reset engine to compute full sequence from scratch for comparison.
    engine.reset()
    output_full = model(t_full, cache_engine=engine)
    
    # Compare the last hidden state of the full run with the incremental decode output.
    # Logic: Full_Run[..., -1, :] should match Decode_Run[..., 0, :].
    out_full_np = output_full.numpy()[:, -1, :]
    out_decode_np = output_decode.numpy()[:, 0, :]
    
    # Calculate the maximum difference between the two outputs.
    diff = np.abs(out_full_np - out_decode_np).max()
    print(f"Max difference between Full and Incremental: {diff}")
    
    # Check if the numerical difference is within an acceptable floating-point tolerance.
    if diff < 1e-4:
        print("✅ Cache Correctness Verified: Incremental matches Full sequence.")
    else:
        print("❌ Cache Correctness Failed: Output mismatch.")

def test_cache_content():
    # Verify that data is actually stored in the underlying StaticKVCache.
    config = LlamaConfig(hidden_size=64, n_layers=1, n_heads=2)
    engine = CacheEngine(config)
    model = LlamaModel(config)
    model.to("cuda")
    
    # Use a specific sequence to track data.
    input_data = Tensor(np.array([[1, 2, 3]]), device="cuda")
    model(input_data, cache_engine=engine)
    
    # Inspect the first layer's K cache.
    k_cache, v_cache, cur_len = engine.get_view(layer_idx=0)
    
    # The current length should reflect the 3 tokens processed.
    assert cur_len == 3
    # Check if the cache is non-zero (indicating data was written).
    assert np.count_nonzero(k_cache.numpy()) > 0
    print("✅ Cache Content Verified: Data successfully persisted in memory.")

if __name__ == "__main__":
    test_kv_cache_logic()
    test_cache_content()