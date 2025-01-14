#include <iostream>
#include <vector>

int main() {
    // vector size
    int N = 256;

    //initialize A,B
    std::vector<float> A(N), B(N), C(N);

    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f; // Vector A: values 0, 1, 2, ..., N-1 as floats
        B[i] = (i + 1) * 1.0f; // Vector B: values 1, 2, 3, ..., N as floats
    }
/*
    std::cout << "Vector A:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Vector B:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << B[i] << " ";
    }
    std::cout << std::endl;
*/

    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];  // Add corresponding elements of A and B
    }
    std::cout << "Vector C:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;


    return 0;
}

// g++ -o vector_addition vector_addition.cpp