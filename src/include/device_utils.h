#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H

void kmeans_cuda(float* array,
	float* centroids,
	int* labels,
	int N,
	int M,
	int K,
	float THRESHOLD,
	bool ver,
	int max_iters);

#endif