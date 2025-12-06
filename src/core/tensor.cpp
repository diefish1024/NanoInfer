#include "tensor.hpp"
#include "cuda_ops.hpp"
#include <numeric>
#include <iostream>
#include <sstream>
#include <cstring>

std::vector<int> Tensor::compute_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size());
    int current_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = current_stride;
        current_stride *= shape[i];
    }
    return strides;
}

Tensor::Tensor(std::vector<int> shape, Device device)
    : shape(std::move(shape)), device(device) {
    
    this->size = std::accumulate(this->shape.begin(), this->shape.end(), 
                                 1UL, std::multiplies<size_t>());
    
    this->strides = compute_strides(this->shape);

    if (this->device == Device::CPU) {
        this->data = new float[this->size];
        std::fill_n(this->data, this->size, 0.0f);
    } else {
        cuda_malloc(&this->data, this->size);
    }
}

Tensor::~Tensor() {
    if (!data) return;

    if (device == Device::CPU) {
        delete[] data;
    } else {
        cuda_free(data);
    }
}

Tensor::Tensor(Tensor&& other) noexcept 
    : data(other.data), shape(std::move(other.shape)), 
      strides(std::move(other.strides)), size(other.size), device(other.device) {
    other.data = nullptr;
    other.size = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (data) {
            if (device == Device::CPU) delete[] data;
            else cuda_free(data);
        }
        data = other.data;
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        size = other.size;
        device = other.device;
        
        other.data = nullptr;
        other.size = 0;
    }
    return *this;
}

Tensor& Tensor::to_cuda() {
    if (device == Device::CUDA) return *this;

    float* gpu_ptr = nullptr;
    cuda_malloc(&gpu_ptr, size);
    
    cuda_memcpy_h2d(gpu_ptr, data, size);
    
    delete[] data;
    data = gpu_ptr;
    device = Device::CUDA;

    return *this;
}

Tensor& Tensor::to_cpu() {
    if (device == Device::CPU) return *this;

    float* cpu_ptr = new float[size];
    
    cuda_memcpy_d2h(cpu_ptr, data, size);
    
    cuda_free(data);
    data = cpu_ptr;
    device = Device::CPU;

    return *this;
}

Tensor* Tensor::add(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("Shape mismatch in add operation");
    }
    if (device != other.device) {
        throw std::runtime_error("Device mismatch in add operation: CPU vs CUDA");
    }

    Tensor* out = new Tensor(shape, device);

    if (device == Device::CPU) {
        for (size_t i = 0; i < size; ++i) {
            out->data[i] = data[i] + other.data[i];
        }
    } else {
        launch_add_kernel(data, other.data, out->data, size);
    }
    return out;
}

Tensor* Tensor::mul(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("Shape mismatch in mul operation");
    }
    if (device != other.device) {
        throw std::runtime_error("Device mismatch in mul operation");
    }

    Tensor* out = new Tensor(shape, device);

    if (device == Device::CPU) {
        for (size_t i = 0; i < size; ++i) {
            out->data[i] = data[i] * other.data[i];
        }
    } else {
        launch_mul_kernel(data, other.data, out->data, size);
    }
    return out;
}

Tensor* Tensor::matmul(const Tensor& other, bool trans_a, bool trans_b) const {
    if (device != Device::CUDA || other.device != Device::CUDA) {
        throw std::runtime_error("MatMul requires both Tensors to be on CUDA.");
    }
    
    int K_a;
    int M;
    
    if (trans_a) {
        if (shape.size() != 2) {
            throw std::runtime_error("Transposed MatMul (trans_a=true) only supports 2D inputs.");
        }
        M = shape[1];
        K_a = shape[0];
    } else {
        K_a = shape.back();
        M = size / K_a;
    }

    if (other.shape.size() != 2) {
        throw std::runtime_error("Weight tensor (B) must be 2D.");
    }
    
    int K_b = trans_b ? other.shape[1] : other.shape[0];
    int N   = trans_b ? other.shape[0] : other.shape[1];

    if (K_a != K_b) {
        throw std::runtime_error("MatMul dimension mismatch: Inner dimensions must match.");
    }
    int K = K_a;

    std::vector<int> out_shape;
    if (trans_a) {
        out_shape = {M, N};
    } else {
        out_shape = shape; 
        out_shape.back() = N;
    }
    
    Tensor* out = new Tensor(out_shape, Device::CUDA);

    int lda = shape.back();
    int ldb = other.shape.back(); 
    int ldc = N;

    cublas_sgemm_wrapper(
        M, N, K,
        data, lda,
        other.data, ldb,
        out->data, ldc,
        trans_a, trans_b
    );

    return out;
}

std::string Tensor::to_string() const {
    std::stringstream ss;
    ss << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i] << (i < shape.size() - 1 ? ", " : "");
    }
    ss << "], device=" << (device == Device::CPU ? "CPU" : "CUDA") << ")";
    return ss.str();
}

void Tensor::print() const {
    std::cout << to_string() << std::endl;
}