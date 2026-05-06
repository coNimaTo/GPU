#include <cuda_runtime.h>
#include <fstream>
#include <cmath>
#include <stdexcept>

#include "utils/CudaUtils.cuh"   // always use paths relative to src/

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

static void saveHeightMap(const HeightMap& hm, const char* path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Could not open output file");
    // Header: grid size as int
    f.write(reinterpret_cast<const char*>(&hm.size), sizeof(int));
    // Data: N*N floats
    f.write(reinterpret_cast<const char*>(hm.data),
            hm.size * hm.size * sizeof(float));
}

// ------------------------------------------------------------------------
// Plano
// ------------------------------------------------------------------------
__global__ // Classic one-task-many-times kernel 
void kernel_plane(HeightMap hm, int c) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hm.size || j >= hm.size) return;
    hm(i, j) = c;
}

void plane(HeightMap &hm, int c) {
    // bloques, threads and kernel launch
    dim3 threads(16, 16);
    dim3 blocks((hm.size + 15) / 16, (hm.size + 15) / 16);

    kernel_plane<<<blocks, threads>>>(hm, c);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ------------------------------------------------------------------------
// Inverted Pyramid
// ------------------------------------------------------------------------
__global__ // Classic one-task-many-times kernel 
void kernel_pyramid(HeightMap hm) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hm.size || j >= hm.size) return;
    hm(i, j) = (float)(abs(i-hm.size/2)+abs(j-hm.size/2))/hm.size;
}

void pyramid(HeightMap &hm) {
    // bloques, threads and kernel launch
    dim3 threads(16, 16);
    dim3 blocks((hm.size + 15) / 16, (hm.size + 15) / 16);

    kernel_pyramid<<<blocks, threads>>>(hm);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ------------------------------------------------------------------------
// V
// ------------------------------------------------------------------------
__global__ // Classic one-task-many-times kernel 
void kernel_V(HeightMap hm) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hm.size || j >= hm.size) return;
    hm(i, j) = (float)abs(i-hm.size/2)/hm.size;
}

void V(HeightMap &hm) {
    // bloques, threads and kernel launch
    dim3 threads(16, 16);
    dim3 blocks((hm.size + 15) / 16, (hm.size + 15) / 16);

    kernel_V<<<blocks, threads>>>(hm);
    CUDA_CHECK(cudaDeviceSynchronize());
}


#define N 257

int main(int argc, char *argv[]) {
    
    HeightMap hm;
    hm.allocate(N);
    char path[64];
    
    plane(hm, 0);
    snprintf(path, sizeof(path), "TerrainTest/plane_%d.bin", N);
    saveHeightMap(hm, path);

    pyramid(hm);
    snprintf(path, sizeof(path), "TerrainTest/pyramid_%d.bin", N);
    saveHeightMap(hm, path);

    V(hm);
    snprintf(path, sizeof(path), "TerrainTest/V_%d.bin", N);
    saveHeightMap(hm, path);

}