// Erosion and sedimentation
#pragma once
#include "utils/CudaUtils.cuh"
#include "StateVector.cuh"
#include "constants.cuh"
#include <cuda_runtime.h>

__global__ void pass4(
    cudaSurfaceObject_t             T1read,
    cudaSurfaceObject_t             T1write,
    cudaSurfaceObject_t             T3read, // I only need to read the new velocity
    int                             N
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    // Read current Height to calculate height difference and local tilt angle
    float4 cC, cL, cR, cT, cB;
    float dh_x, dh_y;
    surf2Dread(&cC, T1read, x * sizeof(float4), y);     // Center
    // Check for borders
    if (x == 0) {
        surf2Dread(&cR, T1read, (x+1) * sizeof(float4), y);     // Right
        dh_x = (cR.x - cC.x)/dx;
    }
    else if (x >= N-1) {
        surf2Dread(&cL, T1read, (x-1) * sizeof(float4), y);     // Left
        dh_x = (cC.x - cL.x)/dx;
    }
    else {
        surf2Dread(&cL, T1read, (x-1) * sizeof(float4), y);     // Left
        surf2Dread(&cR, T1read, (x+1) * sizeof(float4), y);     // Right
        dh_x = (cR.x - cL.x)/(2*dx);
    } 

    // Same with the Y-axis
    if (y == 0) {
        surf2Dread(&cB, T1read, x * sizeof(float4), (y+1));    // Bottom
        dh_y = (cB.x - cC.x)/dx;
    }
    else if (y >= N-1) {
        surf2Dread(&cT, T1read, x * sizeof(float4), (y-1));     // Top
        dh_y = (cC.x - cT.x)/dx;
    }
    else {
        surf2Dread(&cT, T1read, x * sizeof(float4), (y-1));    // Top
        surf2Dread(&cB, T1read, x * sizeof(float4), (y+1));    // Bottom
        dh_y = (cB.x - cT.x)/(2*dx);
    }

    // Read Velocity
    float2 vC;
    surf2Dread(&vC, T3read, x * sizeof(float2), y);

    // Calculate sediment capacity
    float4 new_cC = cC;
    float mod_vel   = sqrtf((vC.x*vC.x+vC.y*vC.y));
    float C = 0;
    if (mod_vel > 1) {
        float sumdh2    = dh_x*dh_x + dh_y*dh_y;
        float sin_alpha = fmaxf(sqrtf(sumdh2/(sumdh2+dx*dx)), 0.01f);
        
        C = Kc * sin_alpha * mod_vel * cC.y;
    }
    
    if (C > cC.z) { // cC.z == sedimento suspendido
        float dSediment = fminf(C - cC.z, cC.x); // Limit for erosion (something like bedrock)
        new_cC.x -= dt * Ks * dSediment;
        new_cC.z += dt * Ks * dSediment;
    }
    else {
        float dSediment = fminf(cC.z - C, cC.z); // Limit for deposition
        new_cC.x += dt * Kd * dSediment;
        new_cC.z -= dt * Kd * dSediment;
    }


    surf2Dwrite(new_cC, T1write, x * sizeof(float4), y);
}

void launchPass4(SimState& state)
{
    int N = state.N;
    pass4<<<grid(N), BLOCK>>>(
        state.T1.readSurf(),
        state.T1.writeSurf(),
        state.T3.readSurf(),
        N
    );
    state.T1.swap();
}