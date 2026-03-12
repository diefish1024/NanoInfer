# NanoInfer

**NanoInfer** is a lightweight, high-performance LLM inference engine built from scratch. It designed to run open-source models like Llama.

It features a hybrid backend architecture that dynamically dispatches operators to Triton, CUDA, or SIMD optimized CPU kernels.

## Key Features

- **Hybrid Backend Architecture**:
    - **Triton**: Used for memory-bound operators (RMSNorm, SiLU, RoPE, Softmax).
    - **CUDA/cuBLAS**: Used for compute-bound operators.
    - **CPU Fallback**: Hand-written SIMD intrinsics for low-latency CPU inference.
- **PyTorch-like API**: Implements a familiar `nn.Module` and `functional` API for easy model building and loading.
- **KV Cache & Attention**:
    - Static and paged KV cache for Llama-style attention.
    - Custom Triton kernels for reshape-and-cache, paged attention, and causal masking.

## Project Structure

```text
NanoInfer/
├── nano_infer/           # Python Frontend
│   ├── nn/               # PyTorch-like Modules (Linear, LlamaMLP)
│   ├── ops/              # Operator Dispatcher (Route to CPU/CUDA/Triton)
│   ├── kernels/          # Custom OpenAI Triton Kernels
│   └── backend/          # Compiled C++ Shared Library (.so)
├── src/                  # C++ Backend
│   ├── core/             # Tensor Class & Memory Allocator
│   ├── binding.cpp       # Pybind11 Python Bindings
│   └── kernels/          # Native Kernels
│       ├── cuda/         # cuBLAS Wrappers
│       └── cpu/          # Naive & AVX2 Implementations
└── tests/                # Unit Tests & Numerical Validation
```

## Build & Installation

### Prerequisites

- CMake \>= 3.20
- CUDA Toolkit (nvcc)
- Python \>= 3.10
- Pybind11

### Compilation

You can install the package in editable mode:

```bash
# 1. Clone the repository
git clone https://github.com/diefish1024/NanoInfer.git
cd NanoInfer

# 2. Build C++ Backend & Install Python bindings
pip install -e .

# OR manually via CMake
mkdir build && cd build
cmake ..
make -j
```

### Runtime Dependencies

- Core runtime: numpy, triton, safetensors
- Example/benchmark: transformers (tokenizer), vllm (optional)

## Usage Example

NanoInfer provides a high-level API similar to PyTorch. Here is how to run a Llama MLP block:

```python
import numpy as np
from nano_infer.core import Tensor
from nano_infer.models.llama import LlamaMLP

# 1. Initialize Model
# Hidden=4096, Intermediate=11008 (Llama-2-7b Config)
model = LlamaMLP(hidden_size=4096, intermediate_size=11008)

# 2. Create Input Tensor (on GPU)
x_data = np.random.randn(1, 128, 4096).astype(np.float32)
x = Tensor(x_data).to_cuda()

# 3. Forward Pass
# Internally dispatches to Triton (SiLU) and cuBLAS (Linear)
output = model(x)

print(f"Output Shape: {output.shape}")
# Output: Tensor(shape=[1, 128, 4096], device=CUDA)
```

## End-to-End Inference

```bash
python ./examples/llama_inference.py \
  --model_path ./models/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Hello, my name is" \
  --max_tokens 64
```

## Benchmark

```bash
python ./examples/benchmark_vllm.py \
  --model_path ./models/TinyLlama-1.1B-Chat-v1.0 \
  --prompt_len 128 \
  --batch_size 1 \
  --max_tokens 256 \
  --measure_decode_only
```

The benchmark output includes the following metrics:

- total: End-to-end latency and tokens/s
- decode: Decoding phase latency and tokens/s (when --measure_decode_only is enabled)

If vLLM is installed, the results for vLLM will also be included; otherwise, it will be skipped automatically.

## Testing & Validation

- Unit tests for core kernels (RMSNorm, RoPE, softmax).
- Paged attention vs. static cache numerical consistency tests.
- KV cache correctness checks for both static and paged backends.

## Roadmap

- [x] **Milestone 1: Infrastructure**
    - [x] C++/Python Hybrid Architecture
    - [x] `nn.Module` & Parameter system
    - [x] Operator Dispatcher (Functional API)
    - [x] Llama MLP Block (Linear + SiLU)
- [x] **Milestone 2: Sequence & Attention**
    - [x] Sequence Operator via Triton (RoPE, RMSNorm)
    - [x] KV Cache Management (PagedAttention)
    - [x] High-performance Custom Kernels (ReshapeAndCache, PagedAttention)
- [ ] **Milestone 3: End-to-End Inference**
    - [ ] Weight Loader (HuggingFace Safetensors)
    - [ ] FlashAttention-v2 Implementation (Prefill)
    - [ ] Llama-7b Generation Loop
    - [ ] Continuous Batching Scheduler
- [ ] **Milestone 4: Heterogeneous Computing & Speculative Decoding**
    - [ ] CPU Kernel Optimization (AVX2/AVX512 for Draft Model)
    - [ ] Speculative Scheduler (Draft-Verify Loop)
    - [ ] KV Cache Rewind Support
    - [ ] Heterogeneous KV Cache Offloading (Host <-> Device Swapping)

