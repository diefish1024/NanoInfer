#pragma once
#include <cstddef>
#include <cublas_v2.h> // cuBLAS

void cuda_malloc(float** ptr, size_t size);
void cuda_free(float* ptr);

void cuda_memcpy_h2d(float* d_ptr, const float* h_ptr, size_t size); // Host to Device
void cuda_memcpy_d2h(float* h_ptr, const float* d_ptr, size_t size); // Device to Host

void launch_add_kernel(const float* a, const float* b, float* c, size_t n);
void launch_mul_kernel(const float* a, const float* b, float* c, size_t n);

void cublas_init();
void cublas_shutdown();
void cublas_sgemm_wrapper(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    bool trans_a = false, bool trans_b = false
);