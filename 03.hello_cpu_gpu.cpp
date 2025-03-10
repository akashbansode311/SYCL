#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
    // Create a queue to select the default device (CPU/GPU)
    queue q;

    // Print "Hello, World!" from the CPU
    std::cout << "Hello, World from CPU!\n";

    // Submit kernel to the queue
    q.submit([&](handler &h) {
        // Define a stream buffer for kernel output inside the handler
        sycl::stream out(1024, 256, h);

        h.parallel_for(range<1>(10), [=](id<1> i) {
            out << "Hello, World from GPU! Thread ID: " << i.get(0) << "\n";
        });
    }).wait(); // Wait for kernel execution to complete

    return 0;
}

