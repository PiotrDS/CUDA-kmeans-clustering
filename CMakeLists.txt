cmake_minimum_required(VERSION 3.18)
project(kmeans_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


add_executable(kmeans_cuda src/host/main.cpp)

set_target_properties(kmeans_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)
