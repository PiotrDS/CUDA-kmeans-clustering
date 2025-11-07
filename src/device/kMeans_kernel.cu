#include <cuda_runtime.h>
#include "kmeans_utils.cuh"
#include <cstdio>
#include <cmath>


__global__ void assign_centroid_cuda(float* array, float* centroids,int* labels ,int n, int m, int k) {

    __shared__ float shared_centroids[1024];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;
    
    if (idx < k) {
        for (int j = 0; j < m; j++) {
            shared_centroids[idx * m + j] = centroids[idx * m + j];
        }
    }
    __syncthreads();


    // assign observation to centroids
    float dist_min = INFINITY;
    int min_cluster = 0;
    float dist = 0.0f;
    if (i < n) {
        for (int center = 0; center < k; center++) {
            dist = 0.0f;

            // calculate euclidean distance from centroid
            for (int j = 0; j < m; j++) {
                dist = dist + (array[i * m + j] - shared_centroids[center * m + j]) * (array[i * m + j] - shared_centroids[center * m + j]);
            }

            __syncthreads();

            // printf("for observation %d distance from centroid %d is %f \n", i, center, dist);
           
            // update minimum distance and current cluster
            if (dist < dist_min) {
                dist_min = dist;
                min_cluster = center;
            }
           
            //printf("for observation %d centroid is %d and distance from it is %f \n", i, min_cluster, dist_min);

            // assign centroid to observation
            labels[i] = min_cluster;

        }
    }
    

}


void assign_centroid(float* array, float* centroids, int* labels, int N, int M, int K) {

    int threads = 256;
    // calculate number of blocks
    int blocks = (N + threads - 1) / threads;
    // launch kernel
    assign_centroid_cuda << <blocks, threads >> > (array, centroids,labels, N,M,K);
}
