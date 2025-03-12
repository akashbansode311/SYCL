#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

constexpr int N = 128 * 4; // Total number of elements
constexpr int THREADS_PER_BLOCK = 128;

using namespace sycl;

// SYCL kernel to perform addition and multiplication
void fp32Kernel(queue &q, float *a, float *b, float *c, float *d, int n) {
    q.submit([&](handler &h) {
        h.parallel_for(range<1>(n), [=](id<1> globalThreadId) {
            int id = globalThreadId[0];
            if (id < n) {
                // Perform multiplication
                float mul_result = a[id] * b[id];
                // Perform addition
                d[id] = mul_result + c[id];
            }
        });
    }).wait();
}

void detectGPUs() {
    auto platforms = platform::get_platforms();
    for (const auto &p : platforms) {
        auto devices = p.get_devices();
        for (const auto &d : devices) {
            std::cout << "Device: " << d.get_info<info::device::name>() << "\n";
            std::cout << "  Total Global Memory: " << d.get_info<info::device::global_mem_size>() / (1024 * 1024 * 1024.0) << " GB\n";
            std::cout << "  Number of Compute Units: " << d.get_info<info::device::max_compute_units>() << "\n";
            std::cout << "  Warp Size (Sub-group size): " << d.get_info<info::device::sub_group_sizes>()[0] << "\n";
        }
    }
}

int main() {
    detectGPUs(); // Detect and print GPU properties

    const int size_fp32 = N * sizeof(float);
    std::vector<float> h_a_fp32(N), h_b_fp32(N), h_c_fp32(N), h_d_fp32(N);

    for (int i = 0; i < N; i++) {
        h_a_fp32[i] = (float)i * 1.0f;
        h_b_fp32[i] = (float)i * 2.0f;
        h_c_fp32[i] = (float)i * 3.0f;
    }

    queue q(gpu_selector_v);

    float *d_a_fp32 = malloc_device<float>(N, q);
    float *d_b_fp32 = malloc_device<float>(N, q);
    float *d_c_fp32 = malloc_device<float>(N, q);
    float *d_d_fp32 = malloc_device<float>(N, q);

    q.memcpy(d_a_fp32, h_a_fp32.data(), size_fp32).wait();
    q.memcpy(d_b_fp32, h_b_fp32.data(), size_fp32).wait();
    q.memcpy(d_c_fp32, h_c_fp32.data(), size_fp32).wait();

    // Time measurement
    auto start = std::chrono::high_resolution_clock::now();

    fp32Kernel(q, d_a_fp32, d_b_fp32, d_c_fp32, d_d_fp32, N);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsedTime = stop - start;
    std::cout << "FP32 Multiplication and Addition Time: " << elapsedTime.count() << " ms\n";

    q.memcpy(h_d_fp32.data(), d_d_fp32, size_fp32).wait();

    free(d_a_fp32, q);
    free(d_b_fp32, q);
    free(d_c_fp32, q);
    free(d_d_fp32, q);

    return 0;
}
