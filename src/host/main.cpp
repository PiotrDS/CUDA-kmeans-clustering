#include <iostream>
#include "kmeans_utils.cuh"
#include <cuda_runtime.h>

int main() {
    std::cout << "Hello from CPU!\n";
    hello_from_gpu();
    cudaDeviceSynchronize();
    return 0;
}
