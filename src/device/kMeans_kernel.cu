#include <cuda_runtime.h>
#include "device_utils.h"
#include <cstdio>
#include <cmath>



__global__ void assign_centroid_cuda(float* array, float* centroids,int* labels ,int n, int m, int k, float threshold) {

    __shared__ float shared_centroids[1024];
    __shared__ int shared_count[50];
    __shared__ int delta[1];



    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;


    // copy centroids to shared memory
    if (idx < k) {
        for (int j = 0; j < m; j++) {
            shared_centroids[idx * m + j] = array[idx * m + j];
        }
    }
    __syncthreads();


    // assign observation to centroids
    float dist_min = INFINITY;
    int min_cluster = 0;
    float dist = 0.0f;
    int label;
    delta[0] = n;
    int iter = 0;
    if (i < n) {

        while ((float)delta[0] / (float)n > threshold && iter < 30) {
            
            // reset delta and shared_count
            delta[0] = 0;
            shared_count[i] = 0;

            if (i == 0)
                iter++;
            for (int center = 0; center < k; center++) {
                dist = 0.0f;

                // calculate euclidean distance from centroid
                for (int j = 0; j < m; j++) {
                    dist = dist + (array[i * m + j] - shared_centroids[center * m + j]) * (array[i * m + j] - shared_centroids[center * m + j]);
                }

                __syncthreads();

                // update minimum distance and current cluster
                if (dist < dist_min) {
                    dist_min = dist;
                    min_cluster = center;
                }


                // assign centroid to observation
                if (labels[i] != min_cluster) {
                    atomicAdd(&delta[0],1);
                    labels[i] = min_cluster;
                }

            }
            // reset old centroids
            if (idx < k) {
                for (int j = 0; j < m; j++) {
                    shared_centroids[idx * m + j] = 0;
                }
            }
            __syncthreads();

            // calculate new centroids as mean of observation in each cluster
            for (int j = 0; j < m; j++) {
                label = labels[i];
                atomicAdd(&shared_centroids[label * m + j], array[i * m + j]);
            }

            atomicAdd(&shared_count[label], 1);

            __syncthreads();

            for (int j = 0; j < m; j++) {
                label = labels[i];
                if(shared_count[label]!=0) shared_centroids[label * m + j] = shared_centroids[label * m + j] / shared_count[label];
                centroids[label * m + j] = shared_centroids[label * m + j];
            }
        }
    }
    

}



void assign_centroid(float* array, float* centroids, int* labels, int N, int M, int K, float THRESHOLD) {

    int threads = 256;
    // calculate number of blocks
    int blocks = (N + threads - 1) / threads;
    // launch kernel
    assign_centroid_cuda << <blocks, threads >> > (array, centroids,labels, N,M,K, THRESHOLD);
}
