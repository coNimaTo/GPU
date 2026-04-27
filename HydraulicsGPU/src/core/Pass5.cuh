// Sediment transportation
#pragma once
#include "utils/CudaUtils.cuh"
#include "StateVector.cuh"
#include "constants.cuh"
#include <cuda_runtime.h>

// Interpolation function between (0,a)<->(1,b)
__host__ __device__ static float lerpf(float a, float b, float t) {
    return a + t * (b - a);
}

__global__ void pass5(
    cudaSurfaceObject_t             T1read,
    cudaSurfaceObject_t             T1write,
    cudaSurfaceObject_t             T3read, // I only need to read the new velocity
    int                             N
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    // Read current Sediments
    float4 cC;
    surf2Dread(&cC, T1read, x * sizeof(float4), y);
    float2 vC;
    surf2Dread(&vC, T3read, x * sizeof(float2), y);

    // Look from where should have come the sediment
    float X = (float)x - vC.x * dt;
    float Y = (float)x - vC.y * dt;

    int oX = floor(X);
    int oY = floor(Y);
    float4 cTL, cTR, cBL, cBR;
    surf2Dread(&cTL, T1read, oX     * sizeof(float4), xY);
    surf2Dread(&cTR, T1read, (oX+1) * sizeof(float4), xY);
    surf2Dread(&cBL, T1read, oX     * sizeof(float4), (xY+1));
    surf2Dread(&cBR, T1read, (oX+1) * sizeof(float4), (xY+1));
    // Bilinear interpolation
    cC.z = lerpf(lerpf(cTL.z, cTR.z, (X-oX)),   //top horizontal
                        lerpf(cBL.z, cBR.z, (X-oX)), // bottom
                        (Y-oY)); // vertical blend

    surf2Dwrite(cC, T1write, x * sizeof(float4), y);
}

void launchPass5(SimState& state)
{
    int N = state.N;
    pass5<<<grid(N), BLOCK>>>(
        state.T1.readSurf(),
        state.T1.writeSurf(),
        state.T3.readSurf(),
        N
    );
    state.T1.swap();
}