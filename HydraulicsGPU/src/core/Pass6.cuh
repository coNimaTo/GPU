// Evaporation
#pragma once
#include "utils/CudaUtils.cuh"
#include "StateVector.cuh"
#include "constants.cuh"
#include <cuda_runtime.h>

__global__ void pass6(
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

    cC.y *= 1 - Ke * dt;
    surf2Dwrite(cC, T1write, x * sizeof(float4), y);
}

void launchPass6(SimState& state)
{
    int N = state.N;
    pass6<<<grid(N), BLOCK>>>(
        state.T1.readSurf(),
        state.T1.writeSurf(),
        N
    );
    state.T1.swap();
}