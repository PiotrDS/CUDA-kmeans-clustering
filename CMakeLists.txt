cmake_minimum_required(VERSION 3.20)
project(kmeans_cuda LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()




add_executable(kmeans_cuda
    src/host/main.cpp
    src/device/kmeans_kernel.cu
)

target_compile_options(kmeans_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)


set_target_properties(kmeans_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON )


target_include_directories(kmeans_cuda PRIVATE
    ${PROJECT_SOURCE_DIR}/src/device
    ${PROJECT_SOURCE_DIR}/src/include
)
target_link_libraries(kmeans_cuda PRIVATE ${CUDA_LIBRARIES})


