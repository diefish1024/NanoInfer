import os
import sys
import time
import argparse
import numpy as np
import torch # For tokenizer only
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_infer.core.tensor import Tensor
from nano_infer.models.llama import LlamaForCausalLM, LlamaConfig
from nano_infer.core.kv_cache import PagedCacheEngine
from nano_infer.utils.weight_loader import load_weights_from_hf, load_config_from_hf

def main():
    parser = argparse.ArgumentParser(description="Run Llama inference with NanoInfer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model directory")
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    
    # 1. Load Config
    try:
        config = load_config_from_hf(args.model_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # 2. Initialize Model
    print("Initializing NanoInfer model...")
    model = LlamaForCausalLM(config)
    
    # 3. Load Weights
    print("Loading weights...")
    try:
        load_weights_from_hf(model, args.model_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    model.to("cuda")

    # 4. Initialize Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 5. Initialize Cache Engine
    print("Initializing Paged Attention Cache Engine...")
    cache_engine = PagedCacheEngine(config, block_size=16, max_num_blocks=1024)
    
    # 6. Encode Prompt
    input_ids_np = tokenizer.encode(args.prompt, return_tensors="np") # [1, Seq]
    input_ids = Tensor(input_ids_np.astype(np.int32), device="cuda")
    
    print(f"\nPrompt: {args.prompt}")
    print("Generating...")
    
    start_time = time.time()
    
    # Prefill
    with torch.no_grad(): # Not needed for NanoInfer but good practice if mixed
        logits = model(input_ids, cache_engine)
        # Logits: [1, Seq, Vocab]
        next_token_logits = logits.to_cpu().numpy()[0, -1, :]
        
        # Simple greedy sampling for demo
        next_token_id = np.argmax(next_token_logits)
        
    generated_ids = [int(next_token_id)]
    sys.stdout.write(tokenizer.decode([next_token_id]))
    sys.stdout.flush()
    
    curr_input = Tensor(np.array([[next_token_id]], dtype=np.int32), device="cuda")
    
    # Decode Loop
    for _ in range(args.max_tokens - 1):
        logits = model(curr_input, cache_engine)
        next_token_logits = logits.to_cpu().numpy()[0, -1, :]
        next_token_id = np.argmax(next_token_logits)
        
        generated_ids.append(int(next_token_id))
        sys.stdout.write(tokenizer.decode([next_token_id]))
        sys.stdout.flush()
        
        if next_token_id == tokenizer.eos_token_id:
            break
            
        curr_input = Tensor(np.array([[next_token_id]], dtype=np.int32), device="cuda")

    end_time = time.time()
    print(f"\n\nGeneration finished in {end_time - start_time:.2f}s")
    print(f"Speed: {len(generated_ids) / (end_time - start_time):.2f} tokens/s")

if __name__ == "__main__":
    main()
