#include <vector>
#include "StateVector.cuh"
#include "HeightMap.cuh"

// Upload and initialize: HeightMap (host) → T1 (device)
void StateUpload(SimState& state, const HeightMap& hm) {
    int N = hm.size;

    cudaDeviceSynchronize();

    // float4(b, 0, 0, 0); water and sediment start at zero
    std::vector<float4> tmp(N * N);
    for (int i = 0; i < N * N; ++i)
        tmp[i] = make_float4(hm.data[i], 0.f, 0.f, 0.f);

    // Write into the "read" slot (indicated by cur)
    cudaMemcpy2DToArray(
        state.T1.arr[state.T1.cur],
        0, 0,                          // x/y offset in bytes
        tmp.data(),                    // host src
        N * sizeof(float4),            // host row pitch
        N * sizeof(float4),            // width in bytes
        N,                             // height (rows)
        cudaMemcpyHostToDevice
    );

    // Zero-init flux and velocity (both slots)
    std::vector<float4> zf(N * N, make_float4(0,0,0,0));
    std::vector<float2> zv(N * N, make_float2(0,0));
    for (int i = 0; i < 2; ++i) {
        cudaMemcpy2DToArray(state.T2.arr[i], 0, 0,
            zf.data(), N * sizeof(float4), N * sizeof(float4), N,
            cudaMemcpyHostToDevice);
        cudaMemcpy2DToArray(state.T3.arr[i], 0, 0,
            zv.data(), N * sizeof(float2), N * sizeof(float2), N,
            cudaMemcpyHostToDevice);
    }
}

// Readback: T1.b (device) → HeightMap (host)
void StateRead(const SimState& state, HeightMap& hm, HeightMap& wm) {
    int N = hm.size;

    std::vector<float4> tmp(N * N);
    cudaMemcpy2DFromArray(
        tmp.data(),
        N * sizeof(float4),            // host row pitch
        state.T1.arr[state.T1.cur],
        0, 0,
        N * sizeof(float4),
        N,
        cudaMemcpyDeviceToHost
    );

    cudaDeviceSynchronize();
    for (int i = 0; i < N * N; ++i){
        hm.data[i] = tmp[i].x;          // .x == b (terrain height)
        wm.data[i] = tmp[i].y;          // .y == w (water amount)
    }
        
}