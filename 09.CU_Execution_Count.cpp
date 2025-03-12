#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

constexpr int numCUs = 40;  // Equivalent to number of CUDA SMs
constexpr int blocks = 680; // Total number of work-groups
constexpr int threadsPerBlock = 1; // Work-items per work-group

int main() {
    queue q{default_selector_v};

    // Allocate memory to track the number of blocks per CU (SM equivalent)
    int *d_blockCountPerCU = malloc_device<int>(numCUs, q);
    int h_blockCountPerCU[numCUs] = {0};

    // Initialize device memory to 0
    q.memset(d_blockCountPerCU, 0, numCUs * sizeof(int)).wait();

    // Launch kernel
    q.parallel_for(nd_range<1>(range<1>(blocks * threadsPerBlock), range<1>(threadsPerBlock)), 
                   [=](nd_item<1> item) {
        int cuId = item.get_group(0) % numCUs; // Approximate Compute Unit ID
        atomic_ref<int, memory_order::relaxed, memory_scope::device, access::address_space::global_space>
            atomicCU(d_blockCountPerCU[cuId]);
        atomicCU.fetch_add(1);
    }).wait();

    // Copy results back to host
    q.memcpy(h_blockCountPerCU, d_blockCountPerCU, numCUs * sizeof(int)).wait();

    // Print the number of blocks executed per Compute Unit
    for (int i = 0; i < numCUs; i++) {
        std::cout << "Compute Unit (CU) " << i << " executed " << h_blockCountPerCU[i] << " blocks\n";
    }

    // Free device memory
    free(d_blockCountPerCU, q);

    return 0;
}
