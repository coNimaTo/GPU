#pragma once
#include "utils/CudaUtils.cuh"   // always use paths relative to src/
#include <fstream>
#include <stdexcept>
#include <algorithm> // for std::minmax_element

struct HeightMap {
    float* data = nullptr;
    int    size = 0;

    __host__ __device__ float& operator()(int row, int col)
    {return data[(size_t)row * size + col];}

    __host__ __device__ const float& operator()(int row, int col) const
    {return data[(size_t)row * size + col];}

    void allocate(int n){
    size = n;
    CUDA_CHECK(cudaMallocManaged(&data, (size_t)n * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(data, 0, (size_t)n * n * sizeof(float)));
    }

    void release() {
        if (data) {                        // ← guard against double-free
            CUDA_CHECK(cudaFree(data));
            data = nullptr;
            size = 0;
        }
    }

    void normalize(){
        int n = size*size;
        // Find min and max values
        auto [min_ptr, max_ptr] = std::minmax_element(data, (data+n));
        float min_val = *min_ptr;
        float max_val = *max_ptr;

        if (max_val == min_val) return; // Avoid division by zero

        for (int i=0; i < n; ++i) {
            data[i] = (data[i] - min_val) / (max_val - min_val);
        }
    }

    void loadHeightMap(const char* path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Could not open input file");

        int n = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(int));
        printf("Read n = %d\n", n);
        if (!f || n <= 0) throw std::runtime_error("Invalid heightmap header");
        size = n;

        f.read(reinterpret_cast<char*>(data), (size_t)n * n * sizeof(float));
        if (!f) {
            release();
            throw std::runtime_error("Unexpected EOF reading heightmap data");
        }
    }
};