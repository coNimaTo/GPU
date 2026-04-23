#pragma once
#include "utils/CudaUtils.cuh"
#include "core/HeightMap.cuh"

void terrain_perlin(HeightMap &hm, int octaves, float persistence,
                    float frequency, unsigned seed);

void terrain_diamond_square(HeightMap &hm, float initial_scale, float decay,
                            unsigned seed);