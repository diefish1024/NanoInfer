#include "tensor.hpp"
#include <cpu_kernels.hpp>
#include <cmath>
#include <cstring>

#define CHECK_FLOAT(x) \
    if (x.dtype != DType::Float32) throw std::runtime_error(#x " must be Float32")
#define CHECK_INT(x) \
    if (x.dtype != DType::Int32) throw std::runtime_error(#x " must be Int32")

void silu(const Tensor &x, Tensor &out) {
    CHECK_FLOAT(x);
    CHECK_FLOAT(out);
    float* x_ptr = static_cast<float*>(x.data);
    float* out_ptr = static_cast<float*>(out.data);

    int n = x.size;
    for (int i = 0; i < n; ++i) {
        float val = x_ptr[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-val));
        out_ptr[i] = val * sigmoid;
    }
}

void embedding(const Tensor& idx, const Tensor& w, Tensor& out) {
    CHECK_INT(idx); 
    CHECK_FLOAT(w);
    CHECK_FLOAT(out);

    int n_idx = idx.size;
    int hidden_dim = w.shape[1];

    int32_t* idx_ptr = static_cast<int32_t*>(idx.data);
    
    float* w_ptr = static_cast<float*>(w.data);
    float* out_ptr = static_cast<float*>(out.data);

    for (int i = 0; i < n_idx; ++i) {
        int id_num = idx_ptr[i];
        float* src = w_ptr + id_num * hidden_dim;
        float* dst = out_ptr + i * hidden_dim;
        std::memcpy(dst, src, hidden_dim * sizeof(float));
    }
}

void softmax(const Tensor& x, Tensor& out) {
    CHECK_FLOAT(x);
    CHECK_FLOAT(out);

    int n_cols = x.shape.back();
    int n_rows = x.size / n_cols;
    
    const float* input_ptr = static_cast<const float*>(x.data);
    float* output_ptr = static_cast<float*>(out.data);

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