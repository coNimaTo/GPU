#pragma once
#include "utils/CudaUtils.h"   // always use paths relative to src/

struct HeightMap {
    float *data;
    int    size;

    __host__ __device__ float& operator()(int row, int col);
    __host__ __device__ const float& operator()(int row, int col) const;

    void allocate(int n);
    void release();
};