#pragma once
#include <cuda_runtime.h>

__constant__ float dt;  // delta time in step
__constant__ float g;   // gravity
__constant__ float dx;  // intercell distance

__constant__ float Kc;  // SedimentCapacity cte
__constant__ float Ks;  // sedimentationRate cte
__constant__ float Kd;  // sedimentDeposition cte
__constant__ float Ke;  // Evaporation cte 

inline void uploadConstants(float v_dt, float v_g, float v_dx,
                            float v_Kc, float v_Ks, float v_Kd, float v_Ke) {
    cudaMemcpyToSymbol(dt, &v_dt, sizeof(float));
    cudaMemcpyToSymbol(g,  &v_g,  sizeof(float));
    cudaMemcpyToSymbol(dx, &v_dx, sizeof(float));

    cudaMemcpyToSymbol(Kc, &v_Kc, sizeof(float));
    cudaMemcpyToSymbol(Ks, &v_Ks, sizeof(float));
    cudaMemcpyToSymbol(Kd, &v_Kd, sizeof(float));
    cudaMemcpyToSymbol(Ke, &v_Ke, sizeof(float));
}


// uploadConstants(0.01f, 9.81f, 1.0f / N);