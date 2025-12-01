#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H

void assign_centroid(float* array, 
	float* centroids, 
	int* labels, 
	int N, 
	int M, 
	int K, 
	float THRESHOLD);

#endif