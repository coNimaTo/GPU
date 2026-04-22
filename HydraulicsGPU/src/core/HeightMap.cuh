#pragma once
#include "utils/CudaUtils.h"   // always use paths relative to src/

struct HeightMap {
    float *data;
    int    size;

    __host__ __device__ float& operator()(int row, int col)
    {return data[row * size + col];}

    __host__ __device__ const float& operator()(int row, int col) const
    {return data[row * size + col];}

    void allocate(int n){
    size = n;
    CUDA_CHECK(cudaMallocManaged(&data, (size_t)n * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(data, 0, (size_t)n * n * sizeof(float)));
    }

    void release(){
    CUDA_CHECK(cudaFree(data));
    data = nullptr;
    };
};