#include "kmeans_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

static int N = 10;
static int M = 10;
static int K = 2;

int main() {
    
    float* array = (float*)malloc(N * M * sizeof(float));
    float* centroids = (float*)malloc(K * M * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float val = static_cast<float>(i * N + j * M);
            if (i < K) {
                centroids[i * M + j] = val;
            }
            array[i * M + j] = val;
        }
    }

    float* cuda_array;
    float* cuda_centroids;
    cudaMalloc((void**)&cuda_array,N * M * sizeof(float));
    cudaMalloc((void**)&cuda_centroids,K * M * sizeof(float));

    cudaMemcpy(cuda_array, array, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_centroids, centroids, K * M * sizeof(float), cudaMemcpyHostToDevice);

    assign_centroid(cuda_array, cuda_centroids);
    cudaDeviceSynchronize();

    free(array);
    free(centroids);
    cudaFree(cuda_array);
    cudaFree(cuda_centroids);

    return 0;
}
