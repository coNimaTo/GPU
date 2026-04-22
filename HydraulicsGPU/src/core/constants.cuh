#pragma once
#include <cuda_runtime.h>

__constant__ float dt;
__constant__ float g;
__constant__ float dx;

inline void uploadConstants(float v_dt, float v_g, float v_dx) {
    cudaMemcpyToSymbol(dt, &v_dt, sizeof(float));
    cudaMemcpyToSymbol(g,  &v_g,  sizeof(float));
    cudaMemcpyToSymbol(dx, &v_dx, sizeof(float));
}


// uploadConstants(0.01f, 9.81f, 1.0f / N);