// Evaporation
#pragma once
#include "utils/CudaUtils.cuh"
#include "StateVector.cuh"
#include "constants.cuh"
#include <cuda_runtime.h>

// Interpolation function between (0,a)<->(1,b)
__host__ __device__ static float lerpf(float a, float b, float t) {
    return a + t * (b - a);
}

__global__ void pass3(
    cudaSurfaceObject_t             T1read,
    cudaSurfaceObject_t             T1write,
    int                             N
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    // Read current cell parameters
    float4 cC;
    surf2Dread(&cC, T1read, x * sizeof(float4), y);

    cC.y *= 1 - Ke * dt
    surf2Dwrite(cC, T1write, x * sizeof(float4), y);
}