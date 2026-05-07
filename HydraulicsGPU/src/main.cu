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

static void saveState(const HeightMap& hm, const HeightMap& wm, const HeightMap& sm, int step) {
    char path[64];
    snprintf(path, sizeof(path), "frames/terrain_%05d.bin", step);
    saveHeightMap(hm, path);
    snprintf(path, sizeof(path), "frames/water_%05d.bin", step);
    saveHeightMap(wm, path);
    snprintf(path, sizeof(path), "frames/sediment_%05d.bin", step);
    saveHeightMap(sm, path);
}

#define N 65

int main(int argc, char *argv[]) {
    printf("Main Started\n");

    // ── Simulation parameters ────────────────────────────────────────────────
    constexpr float DT          = 0.01f;
    constexpr float GRAVITY     = 9.81f;
    constexpr float DX          = 1.0f;
    constexpr float KC          = 0.05f;
    constexpr float KS          = 0.05f;
    constexpr float KD          = 0.05f;
    constexpr float KE          = 0.1f;


    constexpr int   N_STEPS     = 5000;
    constexpr int   FREQ_SAVE   = 10;
    constexpr float RAIN_AMOUNT = 0.4f;
    constexpr float RAIN_DROPS  = 50.0f;

    // ── Upload constants───────────────────────────────────────
    uploadConstants(DT, GRAVITY, DX, KC, KS, KD, KE);

    // ── Init RNG ─────────────────────────────────────────────────────────────
    curandStatePhilox4_32_10_t* randStates = allocRandStates(N, 42UL);

    // ── Init heightmap ───────────────────────────────────────────────────────
    HeightMap hm, wm, sm;
    hm.allocate(N);
    wm.allocate(N);
    sm.allocate(N);

    int algorithm = 0;
    if (argc > 1) {
        algorithm = atoi(argv[1]);
    }
    printf("algorithm : %d\n", algorithm);

    switch (algorithm) {
        case 1:
            terrain_diamond_square(hm, 1.0f, 0.6f, 42);
            printf("diamond terrain created");
            break;
        case 2:
            terrain_perlin(hm, 6, 0.5f, 4.0f, 42);
            printf("perlin terrain created");
            break;
        case 3:
            terrain_plane(hm,1);
            printf("plane terrain created");
            break;
        case 4:
            terrain_pyramid(hm);
            printf("inverted pyramid terrain created");
            break;
        case 5:
            terrain_V(hm);
            printf("V terrain created");
            break;
        default:
            terrain_perlin(hm, 6, 0.5f, 4.0f, 42);
            printf("perlin (default) terrain created");
            break;
    }
    printf("HeightMap created\n");
    hm.normalize();
    printf("HeightMap normalized\n");

    // ── Load heightmap to the State ───────────────────────────────────────────────────────
    SimState state;
    state.allocate(hm.size);
    StateUpload(state, hm);

    // Water Sources
    float wlvl = .005;
    std::vector<WaterSource> sources = {
                                  {2, 4, wlvl},
                     {1, 5, wlvl},{2, 5, wlvl},{3, 5, wlvl},
        {0, 4, wlvl},{1, 6, wlvl},{2, 6, wlvl},{3, 6, wlvl},{4, 6, wlvl},
                     {1, 7, wlvl},{2, 7, wlvl},{3, 7, wlvl},
                                  {2, 8, wlvl},
    };
    WaterSource* d_sources = uploadSources(sources);

    // ── Simulation loop ──────────────────────────────────────────────────────
    for (int step = 0; step < N_STEPS; ++step) {
        launchPass1Rain(state, randStates, RAIN_AMOUNT, RAIN_DROPS);
        launchPass1Sources(state, d_sources, sources.size());

        if (step % FREQ_SAVE == 0) {
            // printf("Step %d\n", step);
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
    hmap_print_ascii(hm, 1);
    
    // ── Cleanup ──────────────────────────────────────────────────────────────
    cudaFree(randStates);
    state.release();
    hm.release();
    wm.release();
    sm.release();
    return 0;
}
