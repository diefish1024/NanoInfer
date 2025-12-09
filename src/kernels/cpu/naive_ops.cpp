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

void embedding(const Tensor& idx, const Tensor& w, Tensor& out) {
    int n_idx = idx.size;
    int hidden_dim = w.shape[1];

    float* idx_p = idx.data;
    float* w_p = w.data;
    float* out_p = out.data;

    for (int i = 0; i < n_idx; ++i) {
        int id_num = static_cast<int>(idx_p[i]);
        float* src = w_p + id_num * hidden_dim;
        float* dst = out_p + i * hidden_dim;
        std::memcpy(dst, src, hidden_dim * sizeof(float));
    }
}