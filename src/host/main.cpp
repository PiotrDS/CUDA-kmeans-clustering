#include "kmeans_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

static int N = 10;
static int M = 5;
static int K = 3;

int main() {
    
    float* array = (float*)malloc(N * M * sizeof(float));
    float* centroids = (float*)malloc(K * M * sizeof(float));
    int* labels = (int*)malloc(N * sizeof(int));

    // test data
    float source[50] = {
        // cluster 1: 
        25.1f, 18.5f, 1.2f, 8.5f, 35.0f,  
        30.5f, 22.0f, 1.0f, 5.2f, 42.5f,  
        22.4f, 16.2f, 1.5f, 10.1f, 33.1f,
        // cluster 2:
        8.2f, 4.1f, 3.5f, 15.2f, 12.5f,   
        11.0f, 5.5f, 3.8f, 12.1f, 14.0f, 
        9.5f, 6.0f, 3.2f, 13.5f, 11.5f,   
        7.5f, 3.9f, 4.0f, 16.0f, 10.9f,   
        // cluster 3: 
        4.1f, 10.5f, 2.5f, 28.5f, 18.0f,  
        6.5f, 12.1f, 2.2f, 25.2f, 20.5f,  
        5.0f, 9.8f, 2.7f, 30.1f, 19.1f   
    };

    for (int i = 0; i < N*M; ++i) {
        array[i] = source[i];
    }

    // copy data to centroids
    for (int j = 0; j < M; j++) {
        centroids[j] = array[j];
        centroids[M + j] = array[3 * M + j];
        centroids[2 * M + j] = array[7 * M + j];
    }
    //for (int i = 0; i < K; i++) {
    //    for (int j = 0; j < M; j++) {
    //        float val = static_cast<float>(i * N + j * M);
    //        if (i < K) {
    //            centroids[i * M + j] = val;
    //        }
    //        array[i * M + j] = val;
    //    }
    //}

    // allocate memery on GPU

    float* cuda_array;
    float* cuda_centroids;
    int* cuda_labels;
    cudaMalloc((void**)&cuda_array,N * M * sizeof(float));
    cudaMalloc((void**)&cuda_centroids,K * M * sizeof(float));
    cudaMalloc((void**)&cuda_labels, N * sizeof(int));

    // copy data from host to device

    cudaMemcpy(cuda_array, array, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_centroids, centroids, K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_labels, labels, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    assign_centroid(cuda_array, cuda_centroids, cuda_labels,N,M,K);


    // copy data from host to device

    cudaMemcpy(array, cuda_array, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids, cuda_centroids, K * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(labels, cuda_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        printf("klaster obserwacji %d to: %d\n", i, labels[i]);
    }

    // free memory
    free(array);
    free(centroids);
    free(labels);
    cudaFree(cuda_array);
    cudaFree(cuda_centroids);
    cudaFree(cuda_labels);
    
    return 0;
}
