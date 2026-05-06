#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>

#include "core/HeightMap.cuh"
#include "core/TerrainGen.cuh"
#include "core/StateVector.cuh"
#include "core/constants.cuh"

#include "core/Pass1.cuh"
#include "core/Pass2.cuh"
#include "core/Pass3.cuh"
#include "core/Pass4.cuh"
#include "core/Pass5.cuh"
#include "core/Pass6.cuh"

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

static void loadHeightMap(HeightMap& hm, const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Could not open input file");
    int N;
    f.read(reinterpret_cast<char*>(&N), sizeof(int));
    hm.allocate(N);
    f.read(reinterpret_cast<char*>(hm.data), N * N * sizeof(float));
}

static void saveState(const HeightMap& hm, const HeightMap& wm, const HeightMap& sm, int step) {
    char path[64];
    snprintf(path, sizeof(path), "frames/terrain_%05d.bin", step);
    saveHeightMap(hm, path);
    snprintf(path, sizeof(path), "frames/water_%05d.bin", step);
    saveHeightMap(wm, path);
    snprintf(path, sizeof(path), "frames/sediment_%05d.bin", step);
    saveHeightMap(sm, path);
}

#define N 257

int main(int argc, char *argv[]) {

    // ── Simulation parameters ────────────────────────────────────────────────
    constexpr float DT          = 0.001f;
    constexpr float GRAVITY     = 9.81f;
    constexpr float DX          = 1.0f;
    constexpr float KC          = 0.01f;
    constexpr float KS          = 0.02f;
    constexpr float KD          = 0.03f;
    constexpr float KE          = 0.01f;


    constexpr int   N_STEPS     = 5000;
    constexpr int   FREQ_SAVE   = 10;
    constexpr float RAIN_AMOUNT = 0.1f;
    constexpr float RAIN_DROPS  = 500.0f;

    // ── Upload constants───────────────────────────────────────
    uploadConstants(DT, GRAVITY, DX, KC, KS, KD, KE);

    // ── Init RNG ─────────────────────────────────────────────────────────────
    curandStatePhilox4_32_10_t* randStates = allocRandStates(N, 42UL);

    // ── Init heightmap ───────────────────────────────────────────────────────
    HeightMap hm, wm, sm;
    hm.allocate(N);
    wm.allocate(N);
    sm.allocate(N);

    // int algorithm = 0;
    // if (argc > 1) {
    //     algorithm = atoi(argv[1]);
    // }

    // switch (algorithm) {
    //     case 1:
    //         terrain_diamond_square(hm, 1.0f, 0.6f, 42);
    //         break;
    //     case 2:
    //         terrain_perlin(hm, 6, 0.5f, 4.0f, 42);
    //         break;
    //     default:
    //         terrain_perlin(hm, 6, 0.5f, 4.0f, 42);
    //         break;
    // }
    // printf("HeightMap created");
    // printf("HeightMap normalization start");
    // hm.normalize();
    // printf("HeightMap normalized");

    char path[64];
    snprintf(path, sizeof(path), "TerrainTest/plane_%d.bin", N);
    loadHeightMap(hm, path);

    // ── Load heightmap to the State ───────────────────────────────────────────────────────
    SimState state;
    state.allocate(hm.size);
    StateUpload(state, hm);

    // Water Sources
    std::vector<WaterSource> sources = {
        {10, 10, .2f}
    };
    WaterSource* d_sources = uploadSources(sources);

    // ── Simulation loop ──────────────────────────────────────────────────────
    for (int step = 0; step < N_STEPS; ++step) {
        launchPass1Rain(state, randStates, RAIN_AMOUNT, RAIN_DROPS);
        // launchPass1Sources  (state, d_sources, sources.size());

        if (step % FREQ_SAVE == 0) {
            printf("Step %d\n", step);
            cudaDeviceSynchronize();

            // Readback and save
            StateRead(state, hm, wm, sm);
            saveState(hm,wm,sm,step);
        }

        launchPass2         (state);
        launchPass3         (state);
        launchPass4         (state);
        launchPass5         (state);
        launchPass6         (state);

    }

    cudaDeviceSynchronize();
    // Readback and save the last iteration
    StateRead(state, hm, wm, sm);
    saveState(hm,wm,sm,N_STEPS);
    
    // ── Readback ────────────────────────────────────────────────────
    hmap_print_ascii(hm, 8);
    
    // ── Cleanup ──────────────────────────────────────────────────────────────
    cudaFree(randStates);
    state.release();
    hm.release();
    wm.release();
    sm.release();
    return 0;
}
