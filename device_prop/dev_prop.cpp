#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);  // Get the number of devices available

    if (deviceCount == 0) {
        std::cout << "No HIP-capable devices found." << std::endl;
        return 0;
    }

    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t deviceProps;
        hipGetDeviceProperties(&deviceProps, i);  // Get properties for the device

        std::cout << "Device " << i << ": " << deviceProps.name << std::endl;
        std::cout << "Max threads per block: " << deviceProps.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: (" 
                  << deviceProps.maxThreadsDim[0] << ", "
                  << deviceProps.maxThreadsDim[1] << ", "
                  << deviceProps.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Max block dimensions: (" 
                  << deviceProps.maxGridSize[0] << ", "
                  << deviceProps.maxGridSize[1] << ", "
                  << deviceProps.maxGridSize[2] << ")" << std::endl;
        std::cout << "Warp Size: " << deviceProps.warpSize<< std::endl;
        std::cout << "Total global memory (in bytes): " << deviceProps.totalGlobalMem << std::endl;
        std::cout << "Maximum shared memory available per block (in bytes): " << deviceProps.sharedMemPerBlock<< std::endl;
        std::cout << "Total constant memory available (in bytes): " << deviceProps.totalConstMem<< std::endl;

    }

    return 0;
}