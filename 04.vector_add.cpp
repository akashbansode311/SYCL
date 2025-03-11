#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>

#define N (5120 * 10000)  // Size of array

using namespace sycl;

int main() {
    // Number of bytes to allocate for N integers
    size_t bytes = N * sizeof(int);

    // Allocate memory for arrays on host
    int *A = static_cast<int*>(malloc(bytes));
    int *B = static_cast<int*>(malloc(bytes));
    int *C = static_cast<int*>(malloc(bytes));

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        A[i] = 1;
        B[i] = 2;
    }

    // Select default SYCL device queue (CPU/GPU)
    queue q;

    // Allocate device memory using USM (Unified Shared Memory)
    int *d_A = malloc_device<int>(N, q);
    int *d_B = malloc_device<int>(N, q);
    int *d_C = malloc_device<int>(N, q);

    // Copy host data to device
    q.memcpy(d_A, A, bytes).wait();
    q.memcpy(d_B, B, bytes).wait();

    // Define execution configuration
    size_t local_work_size = 128; // Local Work-Group Size (equivalent to Threads Per Block)
    size_t global_work_size = (size_t)ceil((float)N / local_work_size) * local_work_size; // Round up to nearest multiple
    size_t num_work_groups = global_work_size / local_work_size; // Equivalent to Blocks in Grid

    // Launch the kernel
    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(global_work_size), range<1>(local_work_size)),
                       [=](nd_item<1> item) {
                           size_t id = item.get_global_linear_id();
                           if (id < N) {
                               d_C[id] = d_A[id] + d_B[id];
                           }
                       });
    }).wait();

    // Copy result back to host
    q.memcpy(C, d_C, bytes).wait();

    // Verify results
    double tolerance = 1.0e-14;
    for (int i = 0; i < N; i++) {
          if (std::fabs(C[i] - 3) > tolerance)  {
            std::cerr << "\nError: value of C[" << i << "] = " << C[i] << " instead of 3\n";
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

    // Print results
    std::cout << "\n---------------------------\n";
    std::cout << "__SUCCESS__\n";
    std::cout << "---------------------------\n";
    std::cout << "N                   = " << N << "\n";
    std::cout << "Local Work-Group Size = " << local_work_size << "\n";
    std::cout << "Number of Work-Groups = " << num_work_groups << "\n";
    std::cout << "Global Work-Item Count = " << global_work_size << "\n";
    std::cout << "---------------------------\n\n";

    return 0;
}
