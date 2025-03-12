#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

using namespace sycl;

constexpr int N = 1024;

// SYCL kernel for FP64 multiplication and addition
void fp64Kernel(queue &q, double *a, double *b, double *c, double *d, int n) {
    q.submit([&](handler &h) {
        h.parallel_for(range<1>(n), [=](id<1> i) {
            d[i] = a[i] * b[i] + c[i];
        });
    }).wait();
}

// SYCL kernel for FP32 multiplication and addition
void fp32Kernel(queue &q, float *a, float *b, float *c, float *d, int n) {
    q.submit([&](handler &h) {
        h.parallel_for(range<1>(n), [=](id<1> i) {
            d[i] = a[i] * b[i] + c[i];
        });
    }).wait();
}

int main() {
    // Print sizes of FP64 and FP32 variables
    std::cout << "Size of double (FP64): " << sizeof(double) << " bytes\n";
    std::cout << "Size of float (FP32): " << sizeof(float) << " bytes\n";
    std::cout << "Size of int: " << sizeof(int) << " bytes\n";

    queue q{default_selector_v};

    double *h_a_fp64 = new double[N];
    double *h_b_fp64 = new double[N];
    double *h_c_fp64 = new double[N];
    double *h_d_fp64 = new double[N];

    float *h_a_fp32 = new float[N];
    float *h_b_fp32 = new float[N];
    float *h_c_fp32 = new float[N];
    float *h_d_fp32 = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a_fp64[i] = i * 1.0;
        h_b_fp64[i] = i * 2.0;
        h_c_fp64[i] = i * 3.0;

        h_a_fp32[i] = i * 4.0f;
        h_b_fp32[i] = i * 5.0f;
        h_c_fp32[i] = i * 6.0f;
    }

    // Allocate device memory
    double *d_a_fp64 = malloc_device<double>(N, q);
    double *d_b_fp64 = malloc_device<double>(N, q);
    double *d_c_fp64 = malloc_device<double>(N, q);
    double *d_d_fp64 = malloc_device<double>(N, q);

    float *d_a_fp32 = malloc_device<float>(N, q);
    float *d_b_fp32 = malloc_device<float>(N, q);
    float *d_c_fp32 = malloc_device<float>(N, q);
    float *d_d_fp32 = malloc_device<float>(N, q);

    // Copy data from host to device
    q.memcpy(d_a_fp64, h_a_fp64, N * sizeof(double)).wait();
    q.memcpy(d_b_fp64, h_b_fp64, N * sizeof(double)).wait();
    q.memcpy(d_c_fp64, h_c_fp64, N * sizeof(double)).wait();

    q.memcpy(d_a_fp32, h_a_fp32, N * sizeof(float)).wait();
    q.memcpy(d_b_fp32, h_b_fp32, N * sizeof(float)).wait();
    q.memcpy(d_c_fp32, h_c_fp32, N * sizeof(float)).wait();

    // Measure execution time for FP64 operations
    auto start_fp64 = std::chrono::high_resolution_clock::now();
    fp64Kernel(q, d_a_fp64, d_b_fp64, d_c_fp64, d_d_fp64, N);
    auto end_fp64 = std::chrono::high_resolution_clock::now();
    float elapsedTime_fp64 = std::chrono::duration<float, std::milli>(end_fp64 - start_fp64).count();
    std::cout << "FP64 Multiplication and Addition Time: " << elapsedTime_fp64 << " ms\n";

    // Measure execution time for FP32 operations
    auto start_fp32 = std::chrono::high_resolution_clock::now();
    fp32Kernel(q, d_a_fp32, d_b_fp32, d_c_fp32, d_d_fp32, N);
    auto end_fp32 = std::chrono::high_resolution_clock::now();
    float elapsedTime_fp32 = std::chrono::duration<float, std::milli>(end_fp32 - start_fp32).count();
    std::cout << "FP32 Multiplication and Addition Time: " << elapsedTime_fp32 << " ms\n";

    // Copy results back to host
    q.memcpy(h_d_fp64, d_d_fp64, N * sizeof(double)).wait();
    q.memcpy(h_d_fp32, d_d_fp32, N * sizeof(float)).wait();

    // Free device memory
    free(d_a_fp64, q);
    free(d_b_fp64, q);
    free(d_c_fp64, q);
    free(d_d_fp64, q);

    free(d_a_fp32, q);
    free(d_b_fp32, q);
    free(d_c_fp32, q);
    free(d_d_fp32, q);

    // Free host memory
    delete[] h_a_fp64;
    delete[] h_b_fp64;
    delete[] h_c_fp64;
    delete[] h_d_fp64;

    delete[] h_a_fp32;
    delete[] h_b_fp32;
    delete[] h_c_fp32;
    delete[] h_d_fp32;

    return 0;
}
