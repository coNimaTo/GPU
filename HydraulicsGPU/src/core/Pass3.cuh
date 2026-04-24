#pragma once
#include "utils/CudaUtils.cuh"
#include "StateVector.cuh"
#include "constants.cuh"
#include <cuda_runtime.h>

__global__ void pass3(
    cudaSurfaceObject_t             T1read,
    cudaSurfaceObject_t             T1write,
    cudaSurfaceObject_t             T2read,
    cudaSurfaceObject_t             T3write, // I only need to write the new velocity
    int                             N
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    // Read current Height, fluxes and velocity
    // Only need height and velocity from the current cell
    float4 cC;
    surf2Dread(&cC, T1read, x * sizeof(float4), y);     // Center
    // Center, Left, Right, Top, Bottom
    float4 fC, fL, fR, fT, fB;
    surf2Dread(&fC, T2read, x * sizeof(float4), y);
    // Check for borders, in that case -> no outflow -> sink
    if (x > 0)   surf2Dread(&fL, T2read, (x-1) * sizeof(float4), y);    // Left
    else         fL = make_float4(0.f, 0.f, 0.f, 0.f);
    if (x < N-1) surf2Dread(&fR, T2read, (x+1) * sizeof(float4), y);    // Right
    else         fR = make_float4(0.f, 0.f, 0.f, 0.f);
    if (y > 0)   surf2Dread(&fT, T2read, x * sizeof(float4), (y-1));    // Top
    else         fT = make_float4(0.f, 0.f, 0.f, 0.f);
    if (y < N-1) surf2Dread(&fB, T2read, x * sizeof(float4), (y+1));    // Bottom
    else         fB = make_float4(0.f, 0.f, 0.f, 0.f);

    // Update waterHeight
    // Delta of water volume in the cell after dt
    float delta_V = dt*((fL.y + fR.x + fT.w + fB.z) // inflow
                        - 
                        (fC.x + fC.y + fC.z + fC.w)); //outflow
    float4 new_cC = cC;
    new_cC.y += delta_V/(dx*dx);

    surf2Dwrite(new_cC, T1write, x * sizeof(float4), y);

    // update velocity
    float2 vC;
    float mean_h = fmaxf((cC.y + new_cC.y)/2, 1e-6f); // Preventing division by zero
    vC.x = ((fL.y + fC.y) - (fC.x + fR.x))/(2*dx*mean_h);
    vC.y =-((fB.z + fC.z) - (fC.w + fT.w))/(2*dx*mean_h); // down direction is positive velocity

    surf2Dwrite(vC, T3write, x * sizeof(float2), y);

}


void launchPass3(SimState& state)
{
    int N = state.N;
    pass3<<<grid(N), BLOCK>>>(
        state.T1.readSurf(),
        state.T1.writeSurf(),
        state.T2.readSurf(),
        state.T3.writeSurf(),
        N
    );
    state.T1.swap();
    state.T3.swap();
}