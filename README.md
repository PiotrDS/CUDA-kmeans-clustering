# Parallel Clustering: GPU-Accelerated K-Means Algorithm 🚀📊

This program implements the **K-Means Clustering Algorithm** with support for both **GPU (CUDA)** and **CPU** execution. It allows you to generate random data, run K-Means algorithm, and save results to CSV files.

---

## Getting Started
### Prerequisites
Download and install the CUDA Toolkit for your corresponding platform. For system requirements and installation instructions of [cuda toolkit](https://developer.nvidia.com/cuda-downloads), please refer to the [Linux Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and the [Windows Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

###Getting the CUDA-kmeans-clustering
Using git clone the repository of CUDA-kmeans-clustering using the command below.
```bash
git clone https://github.com/PiotrDS/CUDA-kmeans-clustering.git
```
###Building CUDA-kmeans-clustering
The CUDA-kmeans-clustering are built using CMake. Follow the instructions below:

Ensure that CMake (version 3.20 or later) is installed. Install it using your package manager if necessary.

Navigate to the root of the cloned repository and create a build directory:
```bash
mkdir build && cd build
```

Configure the project with CMake:
```
cmake ..
cmake --build .
cd Debug
```

Run the program with the following command line arguments:

```bash
kmeans_cuda.exe --n <num_observations> --m <num_dimensions> --k <num_clusters> [options]
```

### Required arguments

* `--n` : Number of observations (N)
* `--m` : Number of dimensions (M)
* `--k` : Number of clusters (K)

### Optional arguments

* `--thr` : Convergence threshold (float, default: 0.1)
* `--v` : Verbose output (bool, default: 0)
* `--s` : Save results to CSV files (bool, default: 0)
* `--g` : Use GPU (bool, default: 1)
* `--ver` : GPU algorithm version (2 different available) (bool, default: 1)
* `--max_iters` : Maximum number of iterations (int, default: 50)

### Example

Run with 1000000 observations, 10 dimensions, 5 clusters on GPU:

```bash
kmeans_cuda.exe --n 1000 --m 3 --k 5 --g 1 --v 1
```

Run the CPU version with verbose output:

```bash
kmeans_cuda.exe --n 500 --m 2 --k 3 --g 0 
```

---

## Output

* If `--v 1` is set, the program prints:

  * Cluster assignment for each observation.
  * Coordinates of all observations.
  * Coordinates of centroids.

* If `--s 1` is set, the program saves:

  * `input_data.csv` : original data
  * `centroids.csv` : centroid coordinates
  * `labels.csv` : cluster assignments

---
