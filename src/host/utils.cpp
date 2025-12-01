#include "host_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <cmath>


// function to generate array of random point drafted from [-10,10) interval
float* generate_random_array(int n, int m) {
    
    float* array = (float*)malloc(n * m * sizeof(float));
    if (!array) {
        fprintf(stderr, "Error: couldn't allocate memory\n");
        return NULL;
    }
    srand((unsigned int)time(NULL));

    for (int i = 0; i < n * m; i++) {
        array[i] = 20 * ((float)rand() / (float)RAND_MAX) - 10; 
    }

    return array;
}

// save centroids to csv file
int save_array_csv(const char* filename,
    const float* array,
    int n,
    int m)
{
    FILE* fp = fopen(filename, "w");
    if (!fp) return 0;

    for (int i = 0; i < n; i++) {
        for (int d = 0; d < m; d++) {
            fprintf(fp, "%g", array[i * m + d]);
            if (d < m - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 1;
}


// save labels to csv file
int save_labels_csv(const char* filename,
    const int* labels,
    int n)
{
    FILE* fp = fopen(filename, "w");
    if (!fp) return 0;

    for (int i = 0; i < n; i++) {
        fprintf(fp, "%d\n", labels[i]);
    }

    fclose(fp);
    return 1;
}


int cpu_kmeans(float* array, float* centroids,int* labels ,int N, int M, int K, float THRESHOLD) {
    
    // initilize centroids as first k observation
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) 
            centroids[i*M + j] = array[i*M + j];
    }

    int* counts = (int*)malloc(K * sizeof(int));

    float dist_min;
    int min_cluster;
    float dist;
    int label;
    int delta = N;
    int iter = 0;
    float sum;
    while ((float)delta / (float)N > THRESHOLD && iter < 1) {

        iter++;
        delta = 0;

        for (int cl = 0; cl < K; cl++) {
            counts[cl] = 0;
        }

        // assign points to centeroid
        for (int i = 0; i < N; i++) {
            dist_min = INFINITY;
            min_cluster = -1;


            for (int c = 0; c < K; c++) {
                sum = 0;
                for (int d = 0; d < M; d++) {
                    float diff = array[i * M + d] - centroids[c * M + d];
                    sum += diff * diff;
                }
                if (sum < dist_min) {
                    dist_min = sum;
                    min_cluster = c;
                }
            }

            counts[min_cluster]++;

            if (labels[i] != min_cluster) {
                labels[i] = min_cluster;
                delta++;
            }
        }

        // calculate new centroids
        for (int c = 0; c < K; c++) {

            // reset centroids
            for (int d = 0; d < M; d++) {
                centroids[c * M + d] = 0;
            }
        }
            
        for (int i = 0; i < N; i++) {
            label = labels[i];
            for (int d = 0; d < M; d++) {
                centroids[label * M + d] += array[i * M + d];
            }
        }
        for (int c = 0; c < K; c++) {
            if (counts[c] == 0) continue;
            for (int d = 0; d < M; d++) centroids[c * M + d] = centroids[c * M + d] / counts[c];
        }
    }

    free(counts);

    return 0;

}
