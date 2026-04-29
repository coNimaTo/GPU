#pragma once
#include "utils/CudaUtils.cuh"   // always use paths relative to src/
#include <algorithm> // for std::minmax_element

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

    void normalize(){
        int n = size*size;
        // Find min and max values
        auto [min_ptr, max_ptr] = std::minmax_element(data, (data+n));
        float min_val = *min_ptr;
        float max_val = *max_ptr;

        if (max_val == min_val) return; // Avoid division by zero

        for (int i=0; i < n; ++n) {
            data[i] = (data[i] - min_val) / (max_val - min_val);
        }
    }
};