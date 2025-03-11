#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>

using namespace sycl;

// Size of array
#define N (5120 * 10000)

int main() {
    // Number of bytes to allocate for N doubles
    size_t bytes = N * sizeof(double);

    // Allocate memory for arrays A, B, and C on host
    double *A = static_cast<double*>(malloc(bytes));
    double *B = static_cast<double*>(malloc(bytes));
    double *C = static_cast<double*>(malloc(bytes));

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    // Create SYCL queue with default device selector (GPU if available)
    queue q(default_selector_v);

    // Allocate memory on the device using USM (Unified Shared Memory)
    double *d_A = malloc_device<double>(N, q);
    double *d_B = malloc_device<double>(N, q);
    double *d_C = malloc_device<double>(N, q);

    // Copy data from host to device
    q.memcpy(d_A, A, bytes).wait();
    q.memcpy(d_B, B, bytes).wait();

    // Define work-group size
    size_t local_work_size = 128;
    size_t global_work_size = ((N + local_work_size - 1) / local_work_size) * local_work_size;
    size_t num_work_groups = global_work_size / local_work_size;

    // Launch kernel
    q.parallel_for(nd_range<1>(range<1>(global_work_size), range<1>(local_work_size)),
                   [=](nd_item<1> item) {
                       size_t id = item.get_global_linear_id();
                       if (id < N) d_C[id] = d_A[id] + d_B[id];
                   }).wait();

    // Copy result back to host
    q.memcpy(C, d_C, bytes).wait();

    // Verify results
    double tolerance = 1.0e-14;
    for (int i = 0; i < N; i++) {
        if (std::fabs(C[i] - 3.0) > tolerance) {
            std::cout << "\nError: value of C[" << i << "] = " << C[i] << " instead of 3.0\n";
            exit(1);
        }
    }

    // Free memory
    free(A);
    free(B);
    free(C);
    free(d_A, q);
    free(d_B, q);
    free(d_C, q);

    // Print success message
    std::cout << "\n---------------------------\n";
    std::cout << "__SUCCESS__\n";
    std::cout << "---------------------------\n";
    std::cout << "N                      = " << N << "\n";
    std::cout << "Local Work-Group Size  = " << local_work_size << "\n";
    std::cout << "Number of Work-Groups  = " << num_work_groups << "\n";
    std::cout << "---------------------------\n\n";

    return 0;
}
