#include "cuda_ops.hpp"
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void cuda_malloc(void** ptr, size_t num_bytes) {
    CUDA_CHECK(cudaMalloc((void**)ptr, num_bytes));
}

void cuda_free(void* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

void cuda_memcpy_h2d(void* d_ptr, const void* h_ptr, size_t num_bytes) {
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, num_bytes, cudaMemcpyHostToDevice));
}

void cuda_memcpy_d2h(void* h_ptr, const void* d_ptr, size_t num_bytes) {
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, num_bytes, cudaMemcpyDeviceToHost));
}

void cuda_memset(void* d_ptr, int value, size_t num_bytes) {
    CUDA_CHECK(cudaMemset(d_ptr, value, num_bytes));
}

__global__ void add_kernel_forward(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
void launch_add_kernel(const float* a, const float* b, float* c, size_t n) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    add_kernel_forward<<<blocks_per_grid, threads_per_block>>>(a, b, c, n);
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mul_kernel_forward(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}
void launch_mul_kernel(const float* a, const float* b, float* c, size_t n) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    mul_kernel_forward<<<blocks_per_grid, threads_per_block>>>(a, b, c, n);
    
    CUDA_CHECK(cudaGetLastError());
}

__global__ void scale_kernel_forward(const float* x, float* y, float alpha, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] * alpha;
    }
}

void launch_scale_kernel(const float* x, float* y, float alpha, size_t n) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    scale_kernel_forward<<<blocks_per_grid, threads_per_block>>>(x, y, alpha, n);

    CUDA_CHECK(cudaGetLastError());
}

static cublasHandle_t cublas_handle = nullptr;

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void cublas_init() {
    if (cublas_handle == nullptr) {
        cudaFree(0);
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
    }
}

void cublas_shutdown() {
    if (cublas_handle != nullptr) {
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        cublas_handle = nullptr;
    }
}

void cublas_sgemm_wrapper(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    bool trans_a, bool trans_b
) {
    if (cublas_handle == nullptr) cublas_init();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // C = A * B  <==>  C^T = B^T * A^T
    // 让 cuBLAS 计算 B^T * A^T，并交换操作符
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    // 1. op_b, op_a (交换操作符)
    // 2. N, M, K (交换维度)
    // 3. B, ldb, A, lda (交换矩阵指针)
    cublasStatus_t status = cublasSgemm(
        cublas_handle,
        op_b, op_a,
        N, M, K,
        &alpha,
        B, ldb,
        A, lda,
        &beta,
        C, ldc
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS SGEMM failed with error code: " + std::to_string(status));
    }
}

void debug_cuda_sync(const char* msg) {
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ CRASH DETECTED at [%s]!\n", msg);
        fprintf(stderr, "👉 Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } else {
        printf("✅ [%s] Clean.\n", msg);
    }
}
