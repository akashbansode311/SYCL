#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    // Get all available SYCL devices
    auto devices = sycl::device::get_devices();

    if (devices.empty()) {
        std::cout << "There is no device supporting SYCL\n";
        return 0;
    }

    std::cout << "There are " << devices.size() << " devices supporting SYCL\n";

    int dev_id = 0;
    for (const auto& device : devices) {
        std::cout << "\nDevice " << dev_id++ << ": \"" << device.get_info<sycl::info::device::name>() << "\"\n";

        std::cout << "  Vendor: " << device.get_info<sycl::info::device::vendor>() << "\n";
        std::cout << "  Driver version: " << device.get_info<sycl::info::device::driver_version>() << "\n";
        std::cout << "  Device type: " 
                  << (device.is_gpu() ? "GPU" : device.is_cpu() ? "CPU" : "Other") 
                  << "\n";
        std::cout << "  Max compute units: " << device.get_info<sycl::info::device::max_compute_units>() << "\n";
        std::cout << "  Max work-group size: " << device.get_info<sycl::info::device::max_work_group_size>() << "\n";
        std::cout << "  Max work-item sizes: ";
        auto work_item_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
        std::cout << work_item_sizes[0] << " x " << work_item_sizes[1] << " x " << work_item_sizes[2] << "\n";

        std::cout << "  Global memory size: " << device.get_info<sycl::info::device::global_mem_size>() << " bytes\n";
        std::cout << "  Local memory size: " << device.get_info<sycl::info::device::local_mem_size>() << " bytes\n";
        std::cout << "  Max memory allocation: " << device.get_info<sycl::info::device::max_mem_alloc_size>() << " bytes\n";
        std::cout << "  Max clock frequency: " << device.get_info<sycl::info::device::max_clock_frequency>() << " MHz\n";
    }

    return 0;
}
