#include <cuda_runtime.h>
#include "kmeans_utils.cuh"
#include <cstdio>

__global__ void assign_centroid_cuda(float* array, float* centroids) {

    int idx = threadIdx.x;
    printf("array[0, %d] = %f\n",idx ,array[idx]);


}

void assign_centroid(float* array, float* centroids) {
    assign_centroid_cuda << <1, 2 >> > (array, centroids);
}
