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

void rope(
    Tensor& q,         // [Batch, Seq, Heads, Dim]
    Tensor& k,         // [Batch, Seq, Heads, Dim]
    const Tensor& cos, // [Max_Seq, Dim/2]
    const Tensor& sin, // [Max_Seq, Dim/2]
    int start_pos
) {
    CHECK_FLOAT(q);
    CHECK_FLOAT(k);
    CHECK_FLOAT(cos);
    CHECK_FLOAT(sin);

    int batch = q.shape[0];
    int seq_len = q.shape[1];
    int heads = q.shape[2];
    int dim = q.shape[3];
    int half_dim = dim / 2;

    float* q_data = static_cast<float*>(q.data);
    float* k_data = static_cast<float*>(k.data);
    const float* cos_data = static_cast<const float*>(cos.data);
    const float* sin_data = static_cast<const float*>(sin.data);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int global_pos = start_pos + s;
            
            const float* cos_row = cos_data + global_pos * half_dim;
            const float* sin_row = sin_data + global_pos * half_dim;

            for (int h = 0; h < heads; ++h) {
                // Offset = b*S*H*D + s*H*D + h*D
                int offset = (b * seq_len * heads * dim) + (s * heads * dim) + (h * dim);
                
                float* q_head = q_data + offset;
                // shape of K may be different to Q, assume they are same now
                float* k_head = k_data + offset;

                // Rotate Half
                for (int i = 0; i < half_dim; ++i) {
                    float c = cos_row[i];
                    float s_val = sin_row[i];

                    float q1 = q_head[i];
                    float q2 = q_head[i + half_dim];
                    q_head[i]            = q1 * c - q2 * s_val;
                    q_head[i + half_dim] = q2 * c + q1 * s_val;

                    float k1 = k_head[i];
                    float k2 = k_head[i + half_dim];
                    k_head[i]            = k1 * c - k2 * s_val;
                    k_head[i + half_dim] = k2 * c + k1 * s_val;
                }
            }
        }
    }
}