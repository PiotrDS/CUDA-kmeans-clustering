#ifndef HOST_UTILS_H
#define HOST_UTILS_H

float* generate_random_array(int n, 
    int m);


int save_array_csv(const char* filename,
    const float* centroids,
    int n,
    int m);

int save_labels_csv(const char* filename,
    const int* labels,
    int n);

int cpu_kmeans(float* array,
    float* centroids,
    int* labels,
    int N,
    int M,
    int K,
    float THRESHOLD);

#endif
