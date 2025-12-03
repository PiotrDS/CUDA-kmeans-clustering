#include "device_utils.h"
#include <cuda_runtime.h>
#include <host_utils.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char** argv) {

    char* centroids_file = NULL;
    char* labels_file = NULL;
    int K = -1;
    double THRESHOLD = 0.1;
    bool gpu = true;
    int v = 0;
    int s = 0;
    int max_iters = 50;
    bool version = true;
    int M, N;

    for (int i = 1; i < argc; i++) {

        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            M = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            K = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--thr") == 0 && i + 1 < argc) {
            THRESHOLD = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--v") == 0 && i + 1 < argc) {
            v = atoi(argv[++i]); 
        }
        else if (strcmp(argv[i], "--s") == 0 && i + 1 < argc) {
            s = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--g") == 0 && i + 1 < argc) {
            gpu = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--ver") == 0 && i + 1 < argc) {
            version = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--max_iters") == 0 && i + 1 < argc) {
            max_iters = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--help") == 0) {
            printf(
                "Usage: kmeans_cuda.exe\n"
                "  --n \t Number of observations\n"
                "  --m \t Number of dimensions\n"
                "  --k \t Number of clusters\n"
                "  --thr \t Threshold [Optional] (float, default: 0.1)\n"
                "  --v \t Verbose [Optional] (bool, default: 0))\n"
                "  --s \t Save results [Optional] (bool, default: 0)\n"
                "  --g \t Use GPU [Optional] (bool, default: 1)\n"
                "  --ver \t Version of GPU algorithm [Optional] (bool, default: 1)\n"
                "  --max_iters \t Maximum iterations [Optional] (int, default: 50)\n"
            );
            return 0;
        }
        else {
            printf("Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }


    
    float* array = generate_random_array(N, M, 1000);
    float* centroids = (float*)malloc(K * M * sizeof(float));
    int* labels = (int*)malloc(N * sizeof(int));

    if (gpu) {

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);   // START POMIARU
        // allocate memery on GPU

        float* cuda_array;
        float* cuda_centroids;
        int* cuda_labels;
        cudaMalloc((void**)&cuda_array, N * M * sizeof(float));
        cudaMalloc((void**)&cuda_centroids, K * M * sizeof(float));
        cudaMalloc((void**)&cuda_labels, N * sizeof(int));

        // copy data from host to device

        cudaMemcpy(cuda_array, array, N * M * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_centroids, centroids, K * M * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_labels, labels, N * sizeof(int), cudaMemcpyHostToDevice);

        // Launch the kernel
        kmeans_cuda(cuda_array, cuda_centroids, cuda_labels, N, M, K, THRESHOLD, version,30);


        // copy data from host to device

        cudaMemcpy(array, cuda_array, N * M * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(centroids, cuda_centroids, K * M * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(labels, cuda_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(cuda_array);
        cudaFree(cuda_centroids);
        cudaFree(cuda_labels);

        cudaEventRecord(stop);  
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        printf("All GPU execution time: %.3f ms\n", ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    else {
        cpu_kmeans(array, centroids, labels, N, M, K, THRESHOLD);
    }

    if (v == 1) {
        for (int i = 0; i < N; i++) {
            printf("Observation number %d is in cluster: %d\n", i, labels[i]);
        }

        for (int i = 0; i < N; i++) {
            printf("Coordinates of observation %d is: [ ", i);
            for (int j = 0; j < M; j++) {
                printf("%f ", array[i * M + j]);
            }
            printf("]\n");
        }

        for (int i = 0; i < K; i++) {
            printf("Coordinates of centroid of cluster %d is: [ ", i);
            for (int j = 0; j < M; j++) {
                printf("%f ", centroids[i * M + j]);
            }
            printf("]\n");
        }
    }

    if (s == 1) {
        save_array_csv("../../input_data.csv", array, N, M);
        save_array_csv("../../centroids.csv", centroids, K, M);
        save_labels_csv("../../labels.csv", labels, N);
    }

    // free memory
    free(array);
    free(centroids);
    free(labels);

    return 0;
}
