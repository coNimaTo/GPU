// Water addition via sources or rain
#pragma once
#include "utils/CudaUtils.cuh"
#include "StateVector.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

static const dim3 BLOCK(16, 16);
static dim3 grid(int N) { return dim3((N+15)/16, (N+15)/16); }

// PHILOX Setup (queda aca xq es el unico lugar donde uso random numbers)
__global__ void setupPhilox(curandStatePhilox4_32_10_t *states, int N, unsigned long long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    // flat index
    int id = y * N + x;
    curand_init(seed, id, 0, &states[id]);
}

// One curandState per cell
curandStatePhilox4_32_10_t* allocRandStates(int N, unsigned long seed) {
    curandStatePhilox4_32_10_t* states;
    cudaMalloc(&states, N * N * sizeof(curandStatePhilox4_32_10_t));
    setupPhilox<<<grid(N), BLOCK>>>(states, N, seed);
    return states;
}

// ── Rain: fixed amount added at random positions ───────────────────

// rainAmount   : water added per raindrop per step
// rainDrops    : mean number of raindrops per steps
__global__ void pass1Rain(
    cudaSurfaceObject_t             T1read,
    cudaSurfaceObject_t             T1write,
    curandStatePhilox4_32_10_t*     states,
    int                 N,
    float               rainAmount,
    float               rainDrops
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    // Read current state
    float4 cell;
    surf2Dread(&cell, T1read, x * sizeof(float4), y);

    // Each thread probabilistically adds water
    // Probability = dt * rainDrops / (N*N) so expected total drops per second = rainDrops
    int idx = y * N + x;
    float r = curand_uniform(&states[idx]);
    if (r < dt * rainDrops / (float)(N * N))
        cell.y += rainAmount;   // .y == d (water height)

    surf2Dwrite(cell, T1write, x * sizeof(float4), y);
}

void launchPass1Rain(SimState& state, curandStatePhilox4_32_10_t* randStates,
                     float rainAmount, float rainDrops)
{
    int N = state.N;
    pass1Rain<<<grid(N), BLOCK>>>(
        state.T1.readSurf(),
        state.T1.writeSurf(),
        randStates,
        N, rainAmount, rainDrops
    );
    state.T1.swap();
}

// ── Water sources: fixed water level held at given positions ─────────────────
struct WaterSource {
    int   x, y;       // cell position
    float level;      // water height to enforce (not add — clamped to this)
};

// sources    : device pointer to array of WaterSource
// nSources   : number of sources
__global__ void pass1Sources(
    cudaSurfaceObject_t T1read,
    cudaSurfaceObject_t T1write,
    int                 N,
    const WaterSource*  sources,
    int                 nSources
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    float4 cell;
    surf2Dread(&cell, T1read, x * sizeof(float4), y);

    // Check if this cell is a source
    for (int i = 0; i < nSources; ++i) {
        if (sources[i].x == x && sources[i].y == y)
            cell.y = fmaxf(cell.y, sources[i].level);  // enforce minimum level
    }

    surf2Dwrite(cell, T1write, x * sizeof(float4), y);
}

WaterSource* uploadSources(const std::vector<WaterSource>& sources) {
    WaterSource* d_sources;
    cudaMalloc(&d_sources, sources.size() * sizeof(WaterSource));
    cudaMemcpy(d_sources, sources.data(),
               sources.size() * sizeof(WaterSource),
               cudaMemcpyHostToDevice);
    return d_sources;  // caller owns this, must cudaFree when done
}

void launchPass1Sources(SimState& state,
                        const WaterSource* d_sources, int nSources)
{
    int N = state.N;
    pass1Sources<<<grid(N), BLOCK>>>(
        state.T1.readSurf(),
        state.T1.writeSurf(),
        N, d_sources, nSources
    );
    state.T1.swap();
}