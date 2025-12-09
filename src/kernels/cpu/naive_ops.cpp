#include "tensor.hpp"
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

void softmax(const Tensor& x, Tensor& out) {
    int n_cols = x.shape.back();
    int n_rows = x.size / n_cols;
    
    const float* input_ptr = x.data;
    float* output_ptr = out.data;

    for (int i = 0; i < n_rows; ++i) {
        const float* row_in = input_ptr + i * n_cols;
        float* row_out = output_ptr + i * n_cols;

        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < n_cols; ++j) {
            if (row_in[j] > max_val) max_val = row_in[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < n_cols; ++j) {
            float exp_val = std::exp(row_in[j] - max_val);
            row_out[j] = exp_val;
            sum += exp_val;
        }

        for (int j = 0; j < n_cols; ++j) {
            row_out[j] /= sum;
        }
    }
}