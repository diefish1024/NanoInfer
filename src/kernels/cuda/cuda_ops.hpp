#pragma once
#include <cstddef>
#include <cublas_v2.h> // cuBLAS

void cuda_malloc(void** ptr, size_t num_bytes);
void cuda_free(void* ptr);

void cuda_memcpy_h2d(void* d_ptr, const void* h_ptr, size_t num_bytes);  // Host to Device
void cuda_memcpy_d2h(void* h_ptr, const void* d_ptr, size_t num_bytes);  // Device to Host
void cuda_memset(void* d_ptr, int value, size_t num_bytes);

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

void debug_cuda_sync(const char* msg);