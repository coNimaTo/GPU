#pragma once
#include "utils/CudaUtils.cuh"
#include "core/HeightMap.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

// ─────────────────────────────────────────────
//  ping-pong buffer
// ─────────────────────────────────────────────
struct SimLayer {
    // Los datos se guardan en el array
    // surface es como un handler para lectura y escritura
    // texture es otro handler que incorpora bilinear interpolation
    //    con la ventaja de que usa unidades de hardware hechas para esto
    cudaArray_t          arr[2]  = {};
    cudaSurfaceObject_t  surf[2] = {};
    cudaTextureObject_t  tex[2]  = {}; // Esto al final no lo use

    // bit para flipear que dato es el actual
    int cur = 0;

    void allocate(int N, cudaChannelFormatDesc desc) {
        for (int i = 0; i < 2; ++i) {
            cudaMallocArray(&arr[i], &desc, N, N,
                            cudaArraySurfaceLoadStore);

            cudaResourceDesc rd{};
            rd.resType         = cudaResourceTypeArray;
            rd.res.array.array = arr[i];

            cudaCreateSurfaceObject(&surf[i], &rd);

            cudaTextureDesc td{};
            td.filterMode      = cudaFilterModeLinear;
            td.addressMode[0]  = cudaAddressModeClamp;
            td.addressMode[1]  = cudaAddressModeClamp;
            td.readMode        = cudaReadModeElementType;
            td.normalizedCoords = 1;

            cudaCreateTextureObject(&tex[i], &rd, &td, nullptr);
        }
    }

    void release() {
        for (int i = 0; i < 2; ++i) {
            cudaDestroySurfaceObject(surf[i]);
            cudaDestroyTextureObject(tex[i]);
            cudaFreeArray(arr[i]);
        }
    }

    // Call after each pass that wrote to cur^1
    void swap() { cur ^= 1; }

    cudaSurfaceObject_t readSurf() const { return surf[cur]; }
    cudaSurfaceObject_t writeSurf()const { return surf[cur ^ 1]; }
    cudaTextureObject_t readTex()  const { return tex[cur]; }
};

// ─────────────────────────────────────────────
//  Simulation state
//  T1: float4(b, d, s, _)      terrain_height/water/sediment
//  T2: float4(fL, fR, fT, fB)  outflow flux
//  T3: float2(u, v)            water velocity
// ─────────────────────────────────────────────
struct SimState {
    SimLayer T1, T2, T3;
    int N = 0;

    void allocate(int n) {
        N = n;
        T1.allocate(n, cudaCreateChannelDesc<float4>());
        T2.allocate(n, cudaCreateChannelDesc<float4>());
        T3.allocate(n, cudaCreateChannelDesc<float2>());
    }

    void release() { T1.release(); T2.release(); T3.release(); }
};

// Upload and initialize: HeightMap (host) → T1 (device)
void StateUpload(SimState& state, const HeightMap& hm);
// Readback: T1.b (device) → HeightMap (host)
void StateRead(const SimState& state, HeightMap& hm, HeightMap& wm, HeightMap& sm);