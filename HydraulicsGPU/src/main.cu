#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>

#include "core/HeightMap.cuh"
#include "core/TerrainGen.cuh"
#include "core/StateVector.cuh"
#include "core/constants.cuh"
#include "core/Pass1.cuh"
#include "core/Pass2.cuh"

#include "utils/Plotter.h"

#define N 257

int main(int argc, char *argv[]) {

    // ── Simulation parameters ────────────────────────────────────────────────
    constexpr float DT          = 0.01f;
    constexpr float GRAVITY     = 9.81f;
    constexpr float DX          = 1.0f;
    constexpr int   N_STEPS     = 1000;
    constexpr float RAIN_AMOUNT = 0.01f;
    constexpr int   RAIN_DROPS  = 512;

    // ── Upload constants───────────────────────────────────────
    uploadConstants(DT, GRAVITY, DX);

    // ── Init RNG ─────────────────────────────────────────────────────────────
    curandStatePhilox4_32_10_t* randStates = allocRandStates(N*N, 42UL);

    // ── Init heightmap ───────────────────────────────────────────────────────
    HeightMap hm;
    hm.allocate(N);

    int algorithm = 0;
    if (argc > 1) {
        algorithm = atoi(argv[1]);
    }

    switch (algorithm) {
        case 1:
            terrain_diamond_square(hm, 1.0f, 0.6f, 42);
            break;
        case 2:
            terrain_perlin(hm, 6, 0.5f, 4.0f, 42);
            break;
        default:
            terrain_perlin(hm, 6, 0.5f, 4.0f, 42);
            break;
    }

    // ── Load heightmap to the State ───────────────────────────────────────────────────────
    SimState state;
    state.allocate(hm.size);
    StateUpload(state, hm);

    // ── Simulation loop ──────────────────────────────────────────────────────
    for (int step = 0; step < N_STEPS; ++step) {
        launchPass1Rain(state, randStates, RAIN_AMOUNT, RAIN_DROPS);
        launchPass2    (state);
    }
    cudaDeviceSynchronize();
    
    // ── Readback ────────────────────────────────────────────────────
    StateRead(state, hm);
    hmap_print_ascii(hm, 8);
    
    // ── Cleanup ──────────────────────────────────────────────────────────────
    cudaFree(randStates);
    state.release();
    hm.release();
    return 0;
}
