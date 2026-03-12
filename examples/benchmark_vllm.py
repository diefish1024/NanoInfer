import os
import sys
import time
import argparse
import numpy as np
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_infer.core.tensor import Tensor
from nano_infer.models.llama import LlamaForCausalLM
from nano_infer.core.kv_cache import PagedCacheEngine
from nano_infer.utils.weight_loader import load_weights_from_hf, load_config_from_hf

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False


def load_nanoinfer(model_path: str, batch_size: int):
    config = load_config_from_hf(model_path)
    config.max_batch_size = max(1, batch_size)
    model = LlamaForCausalLM(config)
    load_weights_from_hf(model, model_path)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return config, model, tokenizer


def build_prompt(tokenizer, prompt: str, prompt_len: int):
    if prompt_len <= 0:
        input_ids = tokenizer.encode(prompt)
        return prompt, input_ids
    base_ids = tokenizer.encode(prompt)
    if len(base_ids) == 0:
        base_ids = [tokenizer.bos_token_id or 0]
    if len(base_ids) >= prompt_len:
        input_ids = base_ids[:prompt_len]
    else:
        input_ids = []
        while len(input_ids) < prompt_len:
            need = prompt_len - len(input_ids)
            input_ids.extend(base_ids[:need])
    return tokenizer.decode(input_ids), input_ids


def run_nanoinfer_once(config, model, tokenizer, input_ids_np: np.ndarray, max_tokens: int, eos_token_id):
    cache_engine = PagedCacheEngine(config, block_size=16, max_num_blocks=1024)
    input_ids = Tensor(input_ids_np, device="cuda")

    prefill_start = time.perf_counter()
    logits = model(input_ids, cache_engine)
    next_token_ids = np.argmax(logits.to_cpu().numpy()[:, -1, :], axis=-1).astype(np.int32)
    prefill_end = time.perf_counter()
    generated = int(next_token_ids.shape[0])
    finished = np.zeros(next_token_ids.shape[0], dtype=bool)
    if eos_token_id is not None:
        finished |= (next_token_ids == eos_token_id)

    curr_input = Tensor(next_token_ids[:, None], device="cuda")
    decode_start = time.perf_counter()
    for _ in range(max_tokens - 1):
        if finished.all():
            break
        active_count = int((~finished).sum())
        logits = model(curr_input, cache_engine)
        next_token_ids = np.argmax(logits.to_cpu().numpy()[:, -1, :], axis=-1).astype(np.int32)
        generated += active_count
        if eos_token_id is not None:
            finished |= (next_token_ids == eos_token_id)
        curr_input = Tensor(next_token_ids[:, None], device="cuda")
    decode_end = time.perf_counter()
    return generated, prefill_end - prefill_start, decode_end - decode_start, decode_end - prefill_start


def load_vllm(
    model_path: str,
    max_model_len: int,
    gpu_mem_util: float,
    attention_backend: str,
    enforce_eager: bool,
    disable_compile: bool,
):
    return LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=enforce_eager,
        **({} if not disable_compile else {"compilation_config": {"mode": 0}}),
        **({} if attention_backend == "auto" else {"attention_config": {"backend": attention_backend}}),
    )


def run_vllm_once(llm, prompts, max_tokens: int):
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    end = time.perf_counter()
    total_tokens = 0
    for out in outputs:
        total_tokens += len(out.outputs[0].token_ids)
    return total_tokens, end - start


def summarize(name: str, token_counts, times, label: str):
    total_tokens = sum(token_counts)
    total_time = sum(times)
    tok_per_s = total_tokens / total_time if total_time > 0 else 0.0
    avg_time = total_time / len(times)
    print(f"{name} | {label} | runs={len(times)} | tokens/s={tok_per_s:.2f} | avg_time={avg_time:.4f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark NanoInfer vs vLLM")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    parser.add_argument("--prompt_len", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure_decode_only", action="store_true")
    parser.add_argument("--skip_nano", action="store_true")
    parser.add_argument("--skip_vllm", action="store_true")
    parser.add_argument("--vllm_max_model_len", type=int, default=1024)
    parser.add_argument("--vllm_gpu_mem_util", type=float, default=0.70)
    parser.add_argument("--vllm_attention_backend", type=str, default="auto")
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_disable_compile", action="store_true")
    args = parser.parse_args()

    prompt_text, prompt_ids = build_prompt(
        AutoTokenizer.from_pretrained(args.model_path), args.prompt, args.prompt_len
    )

    if not args.skip_nano:
        print("Loading NanoInfer model...")
        config, model, tokenizer = load_nanoinfer(args.model_path, args.batch_size)
        eos_token_id = tokenizer.eos_token_id
        input_ids_np = np.array([prompt_ids] * args.batch_size, dtype=np.int32)

        for _ in range(args.warmup):
            run_nanoinfer_once(config, model, tokenizer, input_ids_np, args.max_tokens, eos_token_id)

        nano_tokens = []
        nano_decode_tokens = []
        nano_prefill = []
        nano_decode = []
        nano_total = []
        for _ in range(args.num_runs):
            tks, t_prefill, t_decode, t_total = run_nanoinfer_once(
                config, model, tokenizer, input_ids_np, args.max_tokens, eos_token_id
            )
            nano_tokens.append(tks)
            nano_decode_tokens.append(max(tks - args.batch_size, 0))
            nano_prefill.append(t_prefill)
            nano_decode.append(t_decode)
            nano_total.append(t_total)

        if args.measure_decode_only:
            summarize("NanoInfer", nano_decode_tokens, nano_decode, "decode")
        summarize("NanoInfer", nano_tokens, nano_total, "total")

    if not args.skip_vllm:
        if not VLLM_AVAILABLE:
            print("vLLM 未安装，跳过 vLLM benchmark。")
            return

        print("Loading vLLM model...")
        llm = load_vllm(
            args.model_path,
            args.vllm_max_model_len,
            args.vllm_gpu_mem_util,
            args.vllm_attention_backend,
            args.vllm_enforce_eager,
            args.vllm_disable_compile,
        )
        prompts = [prompt_text] * args.batch_size

        for _ in range(args.warmup):
            run_vllm_once(llm, prompts, args.max_tokens)

        vllm_tokens = []
        vllm_decode_tokens = []
        vllm_total = []
        vllm_decode = []
        for _ in range(args.num_runs):
            tks, t = run_vllm_once(llm, prompts, args.max_tokens)
            vllm_tokens.append(tks)
            vllm_total.append(t)
            if args.measure_decode_only and args.max_tokens > 1:
                _, t_prefill = run_vllm_once(llm, prompts, 1)
                vllm_decode.append(max(t - t_prefill, 1e-9))
                vllm_decode_tokens.append(max(tks - args.batch_size, 0))

        if args.measure_decode_only and len(vllm_decode) == len(vllm_total):
            summarize("vLLM", vllm_decode_tokens, vllm_decode, "decode")
        summarize("vLLM", vllm_tokens, vllm_total, "total")


if __name__ == "__main__":
    main()
