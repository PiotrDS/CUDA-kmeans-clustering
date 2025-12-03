#include <cuda_runtime.h>
#include "device_utils.h"
#include <cstdio>
#include <cmath>


__global__ void assign_centroids(const float* array, const float* centroids, int* labels, int* delta, int n, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;


    float min_dist = INFINITY;
    int min_cluster = 0;
    float dist = 0.0f;

    // assign observation to centroids

    for (int centroid = 0; centroid < k; centroid++) {
        dist = 0.0f;
        for (int j = 0; j < m; j++) {
            // calculate euclidean distance from centroid
            dist += (array[i * m + j] - centroids[centroid * m + j]) * (array[i * m + j] - centroids[centroid * m + j]);
        }
        // update minimum distance and current cluster
        if (dist < min_dist) {
            min_dist = dist;
            min_cluster = centroid;
        }
    }
    // assign centroid to observation
    if (labels[i] != min_cluster) {
        atomicAdd(delta, 1);
        labels[i] = min_cluster;
    }

}

__global__ void sum_centroids_2(const float* array, const int* labels, float* centroid_sums, int* counts, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // calculate sum within each cluster
    int label = labels[i];
    for (int j = 0; j < m; j++) {
        atomicAdd(&centroid_sums[label * m + j], array[i * m + j]);
    }
    //calculate number of observation within each cluster
    atomicAdd(&counts[label], 1);

}

__global__ void sum_centroids(const float* array, const int* labels, float* centroid_sums, int* counts, int n, int m, int k) {
    extern __shared__ float shared_data[];
    // array in shared memory for summing data within blocks
    float* shared_sums = shared_data; 
    // array in shared memory for counting observation in cluster within blocks
    int* shared_counts = (int*)&shared_sums[k * m];

    int idx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // reset arrays
    for (int c = idx; c < k * m; c += blockDim.x) {
        shared_sums[c] = 0.0f;
    }
    for (int c = idx; c < k; c += blockDim.x) {
        shared_counts[c] = 0;
    }
    __syncthreads();

    // add each observation to shared memory
    if (i < n) {
        int label = labels[i];
        for (int j = 0; j < m; j++) {
            atomicAdd(&shared_sums[label * m + j], array[i * m + j]);
        }
        atomicAdd(&shared_counts[label], 1);
    }
    __syncthreads();

    // in each block add summed data from shared memory to global memory
    if (idx == 0) {
        for (int c = 0; c < k; c++) {
            for (int j = 0; j < m; j++) {
                atomicAdd(&centroid_sums[c * m + j], shared_sums[c * m + j]);
            }
            atomicAdd(&counts[c], shared_counts[c]);
        }
    }
}





__global__ void new_centroids(float* centroids, const float* centroid_sums, const int* counts, int k, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    // divide each value by number of observation in corresponding cluster
    for (int j = 0; j < m; j++) {
        if (counts[i] > 0)
            centroids[i * m + j] = centroid_sums[i * m + j] / counts[i];
        else
            centroids[i * m + j] = 0.0f; 
    }

}



void kmeans_cuda(float* array, float* centroids, int* labels, int N, int M, int K, float THRESHOLD, bool ver ,int max_iters) {
    
    int h_delta;
    int threads = 256;
    // calculate number of blocks
    int blocks_points = (N + threads - 1) / threads;
    int blocks_centroids = (K + threads - 1) / threads;
    // launch kernel

    float* d_centroid_sums;
    int* d_counts; 
    int* d_delta;

    cudaMalloc(&d_centroid_sums, K * M * sizeof(float)); 
    cudaMalloc(&d_counts, K * sizeof(int)); 
    cudaMalloc(&d_delta, sizeof(int));

    // Record times 
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);

    cudaEvent_t start_assign, stop_assign;
    cudaEventCreate(&start_assign);
    cudaEventCreate(&stop_assign);

    cudaEvent_t start_sum, stop_sum;
    cudaEventCreate(&start_sum);
    cudaEventCreate(&stop_sum);

    cudaEvent_t start_new, stop_new;
    cudaEventCreate(&start_new);
    cudaEventCreate(&stop_new);

    float time_assign = 0.0f, time_sum = 0.0f, time_new = 0.0f;

    // start total timer
    cudaEventRecord(start_total);

    for (int iter = 0; iter < max_iters; iter++) {
        
        //reset arrays after previous step
        cudaMemset(d_centroid_sums, 0, K * M * sizeof(float));
        cudaMemset(d_counts, 0, K * sizeof(int));
        cudaMemset(d_delta, 0, sizeof(int));
        
        cudaEventRecord(start_assign);
        assign_centroids << <blocks_points, threads >> > (array, centroids, labels, d_delta, N, M, K);
        cudaDeviceSynchronize();
        cudaEventRecord(stop_assign);
        cudaEventSynchronize(stop_assign);

        float ms_assign;
        cudaEventElapsedTime(&ms_assign, start_assign, stop_assign);
        time_assign += ms_assign;

        cudaEventRecord(start_sum);
        if(ver)
            sum_centroids << <blocks_points, threads, K* M * sizeof(float) + K * sizeof(int) >> > (array, labels, d_centroid_sums, d_counts, N, M, K);
        else
            sum_centroids_2 << <blocks_points, threads, K* M * sizeof(float) + K * sizeof(int) >> > (array, labels, d_centroid_sums, d_counts, N, M);
        cudaDeviceSynchronize();
        cudaEventRecord(stop_sum);
        cudaEventSynchronize(stop_sum);

        float ms_sum;
        cudaEventElapsedTime(&ms_sum, start_sum, stop_sum);
        time_sum += ms_sum;

        cudaEventRecord(start_new);
        new_centroids << <blocks_centroids, threads >> > (centroids, d_centroid_sums, d_counts, K, M);
        cudaDeviceSynchronize();
        cudaEventRecord(stop_new);
        cudaEventSynchronize(stop_new);


        float ms_new;
        cudaEventElapsedTime(&ms_new, start_new, stop_new);
        time_new += ms_new;

        cudaMemcpy(&h_delta, d_delta, sizeof(int), cudaMemcpyDeviceToHost);
        if ((float)h_delta / N < THRESHOLD) break;
    }

    // stop total timer
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start_total, stop_total);

    //print results

    printf("\n===== GPU TIMING =====\n");
    printf("assign centroids:        %.3f ms\n", time_assign);
    printf("sum observation:         %.3f ms\n", time_sum);
    printf("calculate new centroids: %.3f ms\n", time_new);
    printf("-------------------------\n");
    printf("Total:                   %.3f ms\n", ms_total);
    printf("=========================\n\n");

    // clear
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_assign);
    cudaEventDestroy(stop_assign);
    cudaEventDestroy(start_sum);
    cudaEventDestroy(stop_sum);
    cudaEventDestroy(start_new);
    cudaEventDestroy(stop_new);
  

}
