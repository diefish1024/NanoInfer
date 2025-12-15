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

Tensor::Tensor(std::vector<int> shape, DType dtype, Device device)
    : shape(std::move(shape)), dtype(dtype), device(device) {
    
    this->size = std::accumulate(this->shape.begin(), this->shape.end(), 
                                 1UL, std::multiplies<size_t>());
    
    this->strides = compute_strides(this->shape);
    size_t bytes = this->nbytes();

    if (this->device == Device::CPU) {
        this->data = new char[bytes];
        std::memset(this->data, 0, bytes);
    } else {
        cuda_malloc(&this->data, bytes);
        cuda_memset(this->data, 0, bytes);
    }
}

Tensor::~Tensor() {
    if (!data) return;
    if (device == Device::CPU) {
        delete[] static_cast<char*>(data);
    } else {
        cuda_free(data);
    }
}

Tensor::Tensor(Tensor&& other) noexcept 
    : data(other.data), shape(std::move(other.shape)), strides(std::move(other.strides)), 
      size(other.size), device(other.device), dtype(other.dtype) {
    other.data = nullptr;
    other.size = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (data) {
            if (device == Device::CPU) delete[] static_cast<char*>(data);
            else cuda_free(data);
        }
        data = other.data;
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        size = other.size;
        device = other.device;
        dtype = other.dtype;
        
        other.data = nullptr;
        other.size = 0;
    }
    return *this;
}

Tensor& Tensor::to_cuda() {
    if (device == Device::CUDA) return *this;

    void* gpu_ptr = nullptr;
    size_t bytes = nbytes();

    cuda_malloc(&gpu_ptr, bytes);
    
    cuda_memcpy_h2d(gpu_ptr, data, bytes);
    
    delete[] static_cast<char*>(data);
    data = gpu_ptr;
    device = Device::CUDA;

    return *this;
}

Tensor& Tensor::to_cpu() {
    if (device == Device::CPU) return *this;

    size_t bytes = nbytes();
    void* cpu_ptr = new char[bytes];
    
    cuda_memcpy_d2h(cpu_ptr, data, bytes);

    cuda_free(data);
    data = cpu_ptr;
    device = Device::CPU;

    return *this;
}

Tensor* Tensor::add(const Tensor& other) const {
    if (shape != other.shape) throw std::runtime_error("Shape mismatch in add");
    if (device != other.device) throw std::runtime_error("Device mismatch");
    if (dtype != other.dtype) throw std::runtime_error("DType mismatch");
    if (dtype != DType::Float32) throw std::runtime_error("Add currently only supports Float32");

    Tensor* out = new Tensor(shape, dtype, device);

    float* a_ptr = static_cast<float*>(data);
    float* b_ptr = static_cast<float*>(other.data);
    float* out_ptr = static_cast<float*>(out->data);

    if (device == Device::CPU) {
        for (size_t i = 0; i < size; ++i) {
            out_ptr[i] = a_ptr[i] + b_ptr[i];
        }
    } else {
        launch_add_kernel(a_ptr, b_ptr, out_ptr, size);
    }
    return out;
}

Tensor* Tensor::mul(const Tensor& other) const {
    if (shape != other.shape) throw std::runtime_error("Shape mismatch");
    if (device != other.device) throw std::runtime_error("Device mismatch");
    if (dtype != other.dtype) throw std::runtime_error("DType mismatch");
    if (dtype != DType::Float32) throw std::runtime_error("Mul currently only supports Float32");

    Tensor* out = new Tensor(shape, dtype, device);

    float* a_ptr = static_cast<float*>(data);
    float* b_ptr = static_cast<float*>(other.data);
    float* out_ptr = static_cast<float*>(out->data);

    if (device == Device::CPU) {
        for (size_t i = 0; i < size; ++i) {
            out_ptr[i] = a_ptr[i] * b_ptr[i];
        }
    } else {
        launch_mul_kernel(a_ptr, b_ptr, out_ptr, size);
    }
    return out;
}

Tensor* Tensor::matmul(const Tensor& other, bool trans_a, bool trans_b) const {
    if (device != Device::CUDA || other.device != Device::CUDA) {
        throw std::runtime_error("MatMul requires both Tensors to be on CUDA.");
    }
    if (dtype != DType::Float32 || other.dtype != DType::Float32) {
        throw std::runtime_error("MatMul only supports Float32");
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
    
    Tensor* out = new Tensor(out_shape, DType::Float32, Device::CUDA);

    int lda = shape.back();
    int ldb = other.shape.back(); 
    int ldc = N;

    cublas_sgemm_wrapper(
        M, N, K,
        static_cast<float*>(data), lda,
        static_cast<float*>(other.data), ldb,
        static_cast<float*>(out->data), ldc,
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
    ss << "], device=" << (device == Device::CPU ? "CPU" : "CUDA") 
       << ", dtype=" << (dtype == DType::Float32 ? "Float32" : "Int32") << ")";
    return ss.str();
}

void Tensor::print() const {
    std::cout << to_string() << std::endl;
}

size_t Tensor::element_size() const {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Int32:   return 4;
        default: return 4;
    }
}

size_t Tensor::nbytes() const {
    return size * element_size();
}