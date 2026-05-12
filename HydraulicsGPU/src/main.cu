#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <chrono>

#include "core/HeightMap.cuh"
#include "core/TerrainGen.cuh"
#include "core/StateVector.cuh"
#include "utils/config.h"
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

// #define N 65

int main(int argc, char *argv[]) {
    printf("Main Started\n");

    const char* configPath = (argc > 2) ? argv[2] : "config.ini";
    Config cfg = Config::load(configPath);
    cfg.print();

    uploadConstants(cfg.DT, cfg.GRAVITY, cfg.DX, cfg.KC, cfg.KS, cfg.KD, cfg.KE);

    int N = cfg.N;
    if (argc > 1) {
        N = atoi(argv[1]);

        if (N <= 0) {
            fprintf(stderr, "Error: N must be > 0\n");
            return 1;
        }
    }

    // ── Init RNG ─────────────────────────────────────────────────────────────
    curandStatePhilox4_32_10_t* randStates = allocRandStates(N, 42UL);

    // ── Init heightmap ───────────────────────────────────────────────────────
    HeightMap hm, wm, sm;
    hm.allocate(N);
    wm.allocate(N);
    sm.allocate(N);

    switch (cfg.algorithm) {
        case 1:
            terrain_diamond_square(hm, 1.0f, 0.6f, 42);
            printf("diamond terrain created\n");
            break;
        case 2:
            terrain_perlin(hm, 6, 0.5f, 4.0f, 42);
            printf("perlin terrain created\n");
            break;
        case 3:
            terrain_plane(hm,1);
            printf("plane terrain created\n");
            break;
        case 4:
            terrain_pyramid(hm);
            printf("inverted pyramid terrain created\n");
            break;
        case 5:
            terrain_V(hm);
            printf("V terrain created\n");
            break;
        case 6:
            terrain_cone(hm);
            printf("cone terrain created\n");
            break;
        default:
            terrain_perlin(hm, 6, 0.5f, 4.0f, 42);
            printf("perlin (default) terrain created\n");
            break;
    }
    // printf("HeightMap created\n");
    hm.normalize();
    // printf("HeightMap normalized\n");

    // ── Load heightmap to the State ───────────────────────────────────────────────────────
    SimState state;
    state.allocate(hm.size);
    StateUpload(state, hm);

    // Water Sources
    float wlvl = cfg.WATER_LEVEL;
    std::vector<WaterSource> sources = {
                                  {2, 4, wlvl},
                     {1, 5, wlvl},{2, 5, wlvl},{3, 5, wlvl},
        {0, 4, wlvl},{1, 6, wlvl},{2, 6, wlvl},{3, 6, wlvl},{4, 6, wlvl},
                     {1, 7, wlvl},{2, 7, wlvl},{3, 7, wlvl},
                                  {2, 8, wlvl},
    };
    WaterSource* d_sources = uploadSources(sources);

    // ── Simulation loop ──────────────────────────────────────────────────────
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < cfg.N_STEPS; ++step) {
        
        // rain and sources stop early so water can evaporate
        if (step < cfg.N_STEPS-cfg.RAIN_STOP) {
            launchPass1Rain(state, randStates, cfg.RAIN_AMOUNT, cfg.RAIN_DROPS);
            // launchPass1Sources(state, d_sources, sources.size());
        }
            

        if (step % cfg.FREQ_SAVE == 0) {
            // printf("Step %d\n", step);
            cudaDeviceSynchronize();

            // Readback and save
            StateRead(state, hm, wm, sm);
            saveState(hm,wm,sm,step);
        }

        launchPass2         (state);
        launchPass3         (state);
        
        // launchPass6         (state);

        launchPass4         (state);
        launchPass5         (state);
        launchPass6         (state);

    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: "
              << elapsed.count()
              << " seconds\n";

    // Readback and save the last iteration
    StateRead(state, hm, wm, sm);
    saveState(hm,wm,sm,cfg.N_STEPS);
    
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
