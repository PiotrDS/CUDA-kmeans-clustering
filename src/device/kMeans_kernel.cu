#include <cuda_runtime.h>
#include "kmeans_utils.cuh"
#include <cstdio>


__global__ void assign_centroid_cuda(float* array, float* centroids, int m, int n, int k) {

    __shared__ float shared_centroids[1024];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;
    
    if (idx < k) {
        for (int j = 0; j < m; j++) {
            shared_centroids[idx * m + j] = centroids[idx * m + j];
        }
    }
    __syncthreads();
    float dist_min = -1.0f;
    int min_cluster = 0;
    float dist = 0.0f;
    if (i < n) {
        for (int center = 0; center < k; center++) {
            dist = 0.0f;
            
            for (int j = 0; j < m; j++) {
                dist = dist + (array[i * m + j] - shared_centroids[center * m + j]) * (array[i * m + j] - shared_centroids[center * m + j]);
            }
            
            if(min_cluster == 0) {
                dist_min = dist;
            }
            else if (dist < dist_min) {
                dist_min = dist;
                min_cluster = center;
            }
        }
        printf("dla obserwacji %d centroidem jest %d \n", i, min_cluster);
    }

}


void assign_centroid(float* array, float* centroids, int M, int N, int K) {

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    assign_centroid_cuda << <blocks, threads >> > (array, centroids, N,M,K);
}
