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

bool Tensor::is_contiguous() const {
    size_t expected_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (shape[i] > 1) {
            if (strides[i] != expected_stride) return false;
            expected_stride *= shape[i];
        }
    }
    return true;
}

// Tensor Tensor::to_contiguous() const {
//     Tensor dst(this->shape_, this->dtype_, this->device_);
    
//     copy_to_contiguous_recursive(dst.data(), this->data(), 0, 0);
    
//     return dst;
// }

void Tensor::reshape(const std::vector<int>& new_shape) {
    size_t new_size = 1;
    for (int dim : new_shape) new_size *= dim;
    
    if (new_size != this->size) {
        throw std::runtime_error("Cannot reshape tensor: total number of elements must remain unchanged.");
    }

    this->shape = new_shape;

    this->strides.resize(new_shape.size());
    size_t s = 1;
    for (int i = new_shape.size() - 1; i >= 0; --i) {
        this->strides[i] = s;
        s *= new_shape[i];
    }
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

Tensor* Tensor::mul_scalar(float alpha) const {
    if (dtype != DType::Float32) {
        throw std::runtime_error("Mul_scalar currently only supports Float32");
    }

    Tensor* out = new Tensor(shape, dtype, device);

    float* a_ptr = static_cast<float*>(data);
    float* out_ptr = static_cast<float*>(out->data);

    if (device == Device::CPU) {
        for (size_t i = 0; i < size; ++i) {
            out_ptr[i] = a_ptr[i] * alpha;
        }
    } else {
        launch_scale_kernel(a_ptr, out_ptr, alpha, size);
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
    const auto& a_shape = shape;
    const auto& b_shape = other.shape;

    if (b_shape.size() == 2) {
        int K_a;
        int M;

        if (trans_a) {
            if (a_shape.size() != 2) {
                throw std::runtime_error("Transposed MatMul (trans_a=true) only supports 2D inputs.");
            }
            M = a_shape[1];
            K_a = a_shape[0];
        } else {
            K_a = a_shape.back();
            M = size / K_a;
        }

        int K_b = trans_b ? b_shape[1] : b_shape[0];
        int N   = trans_b ? b_shape[0] : b_shape[1];

        if (K_a != K_b) {
            throw std::runtime_error("MatMul dimension mismatch: Inner dimensions must match.");
        }

        std::vector<int> out_shape;
        if (trans_a) {
            out_shape = {M, N};
        } else {
            out_shape = a_shape;
            out_shape.back() = N;
        }

        Tensor* out = new Tensor(out_shape, DType::Float32, Device::CUDA);

        int lda = a_shape.back();
        int ldb = b_shape.back();
        int ldc = N;

        cublas_sgemm_wrapper(
            M, N, K_a,
            static_cast<float*>(data), lda,
            static_cast<float*>(other.data), ldb,
            static_cast<float*>(out->data), ldc,
            trans_a, trans_b
        );

        return out;
    }

    if (!trans_a && !trans_b && a_shape.size() >= 3 && b_shape.size() >= 3) {
        if (a_shape.size() != b_shape.size()) {
            throw std::runtime_error("Batched MatMul requires tensors to have the same rank.");
        }
        int ndim = static_cast<int>(a_shape.size());
        for (int i = 0; i < ndim - 2; ++i) {
            if (a_shape[i] != b_shape[i]) {
                throw std::runtime_error("Batched MatMul requires matching leading dimensions.");
            }
        }

        int M = a_shape[ndim - 2];
        int K_a = a_shape[ndim - 1];
        int K_b = b_shape[ndim - 2];
        int N = b_shape[ndim - 1];

        if (K_a != K_b) {
            throw std::runtime_error("MatMul dimension mismatch in batched MatMul: inner dimensions must match.");
        }

        size_t batch_count = 1;
        for (int i = 0; i < ndim - 2; ++i) {
            batch_count *= static_cast<size_t>(a_shape[i]);
        }

        std::vector<int> out_shape = a_shape;
        out_shape[ndim - 1] = N;

        Tensor* out = new Tensor(out_shape, DType::Float32, Device::CUDA);

        float* A_base = static_cast<float*>(data);
        float* B_base = static_cast<float*>(other.data);
        float* C_base = static_cast<float*>(out->data);

        int lda = K_a;
        int ldb = N;
        int ldc = N;

        size_t strideA = static_cast<size_t>(M) * static_cast<size_t>(K_a);
        size_t strideB = static_cast<size_t>(K_b) * static_cast<size_t>(N);
        size_t strideC = static_cast<size_t>(M) * static_cast<size_t>(N);

        for (size_t b = 0; b < batch_count; ++b) {
            const float* A_ptr = A_base + b * strideA;
            const float* B_ptr = B_base + b * strideB;
            float* C_ptr = C_base + b * strideC;

            cublas_sgemm_wrapper(
                M, N, K_a,
                A_ptr, lda,
                B_ptr, ldb,
                C_ptr, ldc,
                false, false
            );
        }

        return out;
    }

    throw std::runtime_error("Unsupported MatMul configuration for given tensor shapes and transpose flags.");
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
