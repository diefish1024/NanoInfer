
import numpy as np
import math
from nano_infer.models.llama import LlamaModel, LlamaConfig
from nano_infer.core.kv_cache import PagedCacheEngine
from nano_infer.core.tensor import Tensor

def test_paged_kv_cache():
    print("Testing Paged KV Cache...")
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        n_layers=2,
        n_heads=4,
        max_position_embeddings=128,
        max_batch_size=1
    )
    
    device = "cuda"
    try:
        model = LlamaModel(config)
        model.to(device)
        # Use PagedCacheEngine
        engine = PagedCacheEngine(config, block_size=16, device=device)
        
        # Prefill
        print("Running Prefill...")
        input_prefill = np.random.randint(0, config.vocab_size, (1, 32))
        t_prefill = Tensor(input_prefill, device=device)
        
        # This should trigger allocation and reshape_and_cache
        output_prefill = model(t_prefill, cache_engine=engine)
        
        assert engine.current_pos == 32
        print(f"Prefill successful. Cache position: {engine.current_pos}")
        
        # Decode
        print("Running Decode...")
        input_decode = np.random.randint(0, config.vocab_size, (1, 1))
        t_decode = Tensor(input_decode, device=device)
        
        # This should trigger paged_attention
        output_decode = model(t_decode, cache_engine=engine)
        
        assert engine.current_pos == 33
        print(f"Decode successful. Cache position: {engine.current_pos}")
        
        print("✅ Paged Cache Test Passed!")
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_paged_kv_cache()
