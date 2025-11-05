#include <cuda_runtime.h>
#include "kmeans_utils.cuh"
#include <cstdio>

__global__ void hello_kernel() {
    printf("Hello from GPU!\n");
}

void hello_from_gpu() {
    hello_kernel << <1, 1 >> > ();
}
