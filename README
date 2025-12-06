# NanoInfer

**NanoInfer** is a lightweight, high-performance LLM inference engine built from scratch. It bridges the gap between **C++ low-level optimization** and **Python high-level flexibility**, designed to run open-source models like Llama.

It features a hybrid backend architecture that dynamically dispatches operators to **OpenAI Triton**, **Native CUDA (cuBLAS)**, or **SIMD optimized CPU** kernels.

## Key Features

- **Hybrid Backend Architecture**:
    - **Triton**: Used for memory-bound operators (RMSNorm, SiLU, RoPE, Softmax).
    - **CUDA/cuBLAS**: Used for compute-bound operators (GEMM/MatMul).
    - **CPU Fallback**: Hand-written **SIMD intrinsics** for low-latency CPU inference.
- **PyTorch-like API**: Implements a familiar `nn.Module` and `functional` API for easy model building andeight loading.

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

You can install the package in editable mode (currently not supported):

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

## Roadmap

- [x] **Milestone 1: Infrastructure**
    - [x] C++/Python Hybrid Architecture
    - [x] `nn.Module` & Parameter system
    - [x] Operator Dispatcher (Functional API)
    - [x] Llama MLP Block (Linear + SiLU)
- [ ] **Milestone 2: Sequence & Attention**
    - [ ] RoPE (Rotary Positional Embeddings) via Triton
    - [ ] KV Cache Management (Static Allocation)
    - [ ] CPU SIMD Optimization
- [ ] **Milestone 3: End-to-End Inference**
    - [ ] Weight Loader (HuggingFace Safetensors)
    - [ ] FlashAttention-v2 Implementation
    - [ ] Llama-7b Generation Loop
