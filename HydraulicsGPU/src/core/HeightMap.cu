#include "HeightMap.h"

__host__ __device__
float& HeightMap::operator()(int row, int col) {
    return data[row * size + col];
}

// Claude dice que es buena practica tener esto para funciones read-only
__host__ __device__
const float& HeightMap::operator()(int row, int col) const {
    return data[row * size + col];
}

void HeightMap::allocate(int n) {
    size = n;
    CUDA_CHECK(cudaMallocManaged(&data, (size_t)n * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(data, 0, (size_t)n * n * sizeof(float)));
}

void HeightMap::release() {
    CUDA_CHECK(cudaFree(data));
    data = nullptr;
}