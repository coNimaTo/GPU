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

// ── Helpers ──────────────────────────────────────────────────────────────────

static void saveHeightMap(const HeightMap& hm, const char* path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Could not open output file");
    // Header: grid size as int
    f.write(reinterpret_cast<const char*>(&hm.size), sizeof(int));
    // Data: N*N floats
    f.write(reinterpret_cast<const char*>(hm.data),
            hm.size * hm.size * sizeof(float));
}

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
    curandStatePhilox4_32_10_t* randStates = allocRandStates(N, 42UL);

    // ── Init heightmap ───────────────────────────────────────────────────────
    HeightMap hm, wm;
    hm.allocate(N);
    wm.allocate(N);

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

    printf("HeightMap created");

    // ── Load heightmap to the State ───────────────────────────────────────────────────────
    SimState state;
    state.allocate(hm.size);
    StateUpload(state, hm);

    // ── Simulation loop ──────────────────────────────────────────────────────
    for (int step = 0; step < N_STEPS; ++step) {
        launchPass1Rain(state, randStates, RAIN_AMOUNT, RAIN_DROPS);
        launchPass2    (state);

        if (step % 50 == 0) {
            printf("Step %d\n", step);
            cudaDeviceSynchronize();

            // Readback and save
            StateRead(state, hm, wm);
            char path[64];
            snprintf(path, sizeof(path), "frames/terrain_%04d.bin", step);
            saveHeightMap(hm, path);
            snprintf(path, sizeof(path), "frames/water_%04d.bin", step);
            saveHeightMap(wm, path);
        }
    }
    cudaDeviceSynchronize();
    
    // ── Readback ────────────────────────────────────────────────────
    StateRead(state, hm, wm);
    hmap_print_ascii(hm, 8);
    
    // ── Cleanup ──────────────────────────────────────────────────────────────
    cudaFree(randStates);
    state.release();
    hm.release();
    wm.release();
    return 0;
}
