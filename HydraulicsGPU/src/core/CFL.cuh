#pragma once
#include <cuda_runtime.h>
#include "constants.cuh"

// Global flag in device memory — 0 = OK, 1 = violated
__device__ int d_CFLviolated;

// Call once at startup
inline void initCFLFlag() {
    int zero = 0;
    cudaMemcpyToSymbol(d_CFLviolated, &zero, sizeof(int));
}

// Call at the start of each step to reset the flag
inline void resetCFLFlag() {
    int zero = 0;
    cudaMemcpyToSymbol(d_CFLviolated, &zero, sizeof(int));
}

// Call after kernels to check — returns true if violated
inline bool checkCFLFlag() {
    int val = 0;
    cudaMemcpyFromSymbol(&val, d_CFLviolated, sizeof(int));
    return val != 0;
}

// Call this inside any kernel where you want to check
__device__ inline void CFLcheck(float vx, float vy) {
    if (fabsf(vx * dt) > dx || fabsf(vy * dt) > dx)
        atomicOr(&d_CFLviolated, 1);
}