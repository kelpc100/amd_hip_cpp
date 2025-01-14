#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#define N 256  // Vector size as preprocessor macro


__global__ void vectorAdd(float* X, float* Y, float* Z, int size) { // This function can work with arguments, but cannot change where arguments points
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Calculates global thread index

    if (i < size) {
        Z[i] = X[i] + Y[i];  // Add corresponding elements of X and Y
    }
}

int main() {
    //initialize A,B
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f; // Vector A: values 0, 1, 2, ..., N-1 as floats
        B[i] = (i + 1) * 1.0f; // Vector B: values 1, 2, 3, ..., N as floats
    }
    
    // Declare pointers for device memory (on the GPU)
    float* d_A; //This is a pointer (*) to a float array (or vector)
    float* d_B; //These are the vectors that will be store on GPU
    float* d_C; //Different from the orginal A, B, and C

    // Allocate memory on the device (GPU)
    hipMalloc((void**)&d_A, N * sizeof(float)); //The vector is still not on the GPU, but memory is allocated for it.
    hipMalloc((void**)&d_B, N * sizeof(float)); //(void**)&d_A is the address of the pointer d_A and needs to be able to modify it
    hipMalloc((void**)&d_C, N * sizeof(float)); //N * sizeof(float) gives the total number of bytes needed to store N elements of type float.

    // Copy data from host (CPU) to device (GPU)
    hipMemcpy(d_A, A.data(), N * sizeof(float), hipMemcpyHostToDevice); //Copy vector A to the GPU as pointer vector d_A of size N * sizeof(float) CPU -> GPU
    hipMemcpy(d_B, B.data(), N * sizeof(float), hipMemcpyHostToDevice); 

    // Define kernel launch parameters
    int blockSize = 256;  // Number of threads per block 
    int numBlocks = (N + blockSize - 1) / blockSize;  // Number of blocks, ensure full coverage of data equation based on N ensures blocks>=needed

    // Launch the kernel to perform vector addition on the GPU
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Copy the result from device (GPU) back to host (CPU)
    hipMemcpy(C.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost);  // Copy result from d_C on GPU to vector C of size N * sizeof(float) GPU -> CPU

    std::cout << "Vector C:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory (GPU) after we're done with it
    hipFree(d_A);  // Free memory allocated for vector A on the GPU
    hipFree(d_B);  // Free memory allocated for vector B on the GPU
    hipFree(d_C);  // Free memory allocated for vector C on the GPU

    return 0;
}
