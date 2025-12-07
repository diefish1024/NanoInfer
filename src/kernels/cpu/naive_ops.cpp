#include <cpu_kernels.hpp>
#include <cmath>
#include <cstring>

void silu(const Tensor &x, Tensor &out) {
    int n = x.size;
    for (int i = 0; i < n; ++i) {
        float val = x.data[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-val));
        out.data[i] = val * sigmoid;
    }
}