#pragma once
#include "utils/CudaUtils.cuh"
#include "StateVector.cuh"
#include "constants.cuh"
#include <cuda_runtime.h>

__global__ void pass2(
    cudaSurfaceObject_t             T1read,
    cudaSurfaceObject_t             T2read,
    cudaSurfaceObject_t             T2write,
    int                             N
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    // Read current state
    // Center, Left, Right, Top, Bottom
    float4 cC, cL, cR, cT, cB;
    float4 cFlux;
    
    surf2Dread(&cC, T1read, x * sizeof(float4), y);     // Center
    float hC = cC.x + cC.y;
    // Check for borders, in that case -> same height -> no outflow
    if (x > 0)   surf2Dread(&cL, T1read, (x-1) * sizeof(float4), y);    // Left
    else         cL = make_float4(hC, 0.f, 0.f, 0.f);
    if (x < N-1) surf2Dread(&cR, T1read, (x+1) * sizeof(float4), y);    // Right
    else         cR = make_float4(hC, 0.f, 0.f, 0.f);
    if (y > 0)   surf2Dread(&cT, T1read, x * sizeof(float4), (y-1));    // Top
    else         cT = make_float4(hC, 0.f, 0.f, 0.f);
    if (y < N-1) surf2Dread(&cB, T1read, x * sizeof(float4), (y+1));    // Bottom
    else         cB = make_float4(hC, 0.f, 0.f, 0.f);

    // Outward Flux in the four directions
    surf2Dread(&cFlux, T2read, x * sizeof(float4), y);

    // Height difference
    // dh_i = (b + d)(x,y) − (b + d)(neighbor_i)
    float4 dh = make_float4(hC - (cL.x + cL.y), 
                            hC - (cR.x + cR.y), 
                            hC - (cT.x + cT.y),
                            hC - (cB.x + cB.y));
    
    // Flux at next step
    // f_i_new = max(0,  f_i + dt · A · g · dh_i / l)
    float4 cNewFlux;
    float k = dt * g * dx; // A = dx*dy, l = dx or dy, dx = dy 
    cNewFlux.x = fmaxf(0.f, cFlux.x + k * dh.x);
    cNewFlux.y = fmaxf(0.f, cFlux.y + k * dh.y);
    cNewFlux.z = fmaxf(0.f, cFlux.z + k * dh.z);
    cNewFlux.w = fmaxf(0.f, cFlux.w + k * dh.w);

    // Scaling if not enough water
    // K = min(1,  d · dx · dy / (sum(f_i_new) · dt))
    // f_i_final = K · f_i_new
    float totalFlux = cNewFlux.x + cNewFlux.y + cNewFlux.z + cNewFlux.w; 
    // Hay que checkear si totalFlux = 0 para no dividir por 0 (lo aprendi por las malas)
    k = (totalFlux > 0.f) ? fminf(1.f, cC.y * dx * dx / (totalFlux * dt)) : 0.f;
    cNewFlux.x *= k; cNewFlux.y *= k; cNewFlux.z *= k; cNewFlux.w *= k;

    surf2Dwrite(cNewFlux, T2write, x * sizeof(float4), y);
}

void launchPass2(SimState& state)
{
    int N = state.N;
    pass2<<<grid(N), BLOCK>>>(
        state.T1.readSurf(),
        state.T2.readSurf(),
        state.T2.writeSurf(),
        N
    );
    state.T2.swap();
}