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
    int gpu = 1;
    int v = 0;
    int s = 0;

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
            v = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--s") == 0 && i + 1 < argc) {
            s = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--g") == 0 && i + 1 < argc) {
            gpu = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: program --n Number of observation --m Number of dimensions --k Number of cluster --thr Threshold --v Verbose --s Save results --g GPU version\n");
            return 0;
        }
        else {
            printf("Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    
    float* array = generate_random_array(N, M);
    float* centroids = (float*)malloc(K * M * sizeof(float));
    int* labels = (int*)malloc(N * sizeof(int));

    if (gpu == 1) {
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
        assign_centroid(cuda_array, cuda_centroids, cuda_labels, N, M, K, THRESHOLD);


        // copy data from host to device

        cudaMemcpy(array, cuda_array, N * M * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(centroids, cuda_centroids, K * M * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(labels, cuda_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        cudaFree(cuda_array);
        cudaFree(cuda_centroids);
        cudaFree(cuda_labels);
    }
    else {
        cpu_kmeans(array, centroids, labels, N, M, K, THRESHOLD);
    }

    if (v == 1) {
        for (int i = 0; i < N; i++) {
            printf("klaster obserwacji %d to: %d\n", i, labels[i]);
        }

        for (int i = 0; i < K; i++) {
            printf("wspo³rzedne %d-tego klastra to: [ ", i);
            for (int j = 0; j < M; j++) {
                printf("%f ", centroids[i * M + j]);
            }
            printf("]\n");
        }
    }

    if (s == 1) {
        save_array_csv("../../input_data.csv", array, N, M);
        save_array_csv("../../centroids.csv", array, K, M);
        save_labels_csv("../../labels.csv", labels, N);
    }

    // free memory
    free(array);
    free(centroids);
    free(labels);

    return 0;
}
