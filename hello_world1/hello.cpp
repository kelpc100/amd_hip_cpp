#include <hip/hip_runtime.h> //where hip_rutime.h is stored aka include folder
#include <iostream> //standard c++ lib

__global__ void hello_world_kernel() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d\n", id);
}
//__global__ sends function to gpu
//void does not return a value
//blockIdx.x gives the index of the block in the grid.
    //A block is a group of threads that execute the same code in parallel on the GPU. The grid is a collection of blocks.
    //blockIdx.x identifies which block a particular thread is a part of, along the x dimension of the grid.
//blockDim.x gives the number of threads per block.
//threadIdx.x gives the thread index within its block.
//int id = blockIdx.x * blockDim.x + threadIdx.x
    //global thread index (or global thread ID) across all blocks in the grid

int main() {
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;

    // Launch the kernel
    hello_world_kernel<<<blocksPerGrid, threadsPerBlock>>>();
    //blocksPerGrid = grid size
        // increase -> increases spread of blocks
    //threadsPerBlock = block size
        //increase -> increase block power


    // Synchronize to ensure the kernel finishes
    hipError_t error = hipDeviceSynchronize();
    //This forces the CPU to wait until the GPU finishes executing all previous operations
    //If the return value is not hipSuccess, an error message is printed,

    // Check for errors during synchronization
    if (error != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(error) << std::endl;
        return -1;  // Return with an error code if synchronization failed
    }

    return 0;
}

// hipcc -o hello hello.cpp
// ./hello