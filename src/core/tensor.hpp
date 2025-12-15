#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <stdexcept>

enum class Device { CPU, CUDA };
enum class DType { Float32, Int32 };

class Tensor {
public:
    void* data;
    DType dtype;

    std::vector<int> shape;
    std::vector<int> strides;
    size_t size;
    Device device;

    Tensor(std::vector<int> shape, DType dtype = DType::Float32, Device device = Device::CPU);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    Tensor& to_cuda();
    Tensor& to_cpu();

    Tensor* add(const Tensor& other) const;
    Tensor* mul(const Tensor& other) const;

    Tensor* matmul(const Tensor& other, bool trans_a = false, bool trans_b = false) const;

    std::string to_string() const;
    void print() const;

    size_t element_size() const;
    size_t nbytes() const;

private:
    static std::vector<int> compute_strides(const std::vector<int>& shape);
};