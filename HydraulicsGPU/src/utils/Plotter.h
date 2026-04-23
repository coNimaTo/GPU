#pragma once
#include "core/HeightMap.cuh"
#include <stdio.h>     // printf

//-------------------------------------------------------
// ASCII preview
//-------------------------------------------------------

void hmap_print_ascii(HeightMap &hm, int downsample) {
    const char *shading = " .:-=+*#%@";
    int n = hm.size;
    float lo = hm(0,0), hi = hm(0,0);
    for (int k = 0; k < n*n; k++) {
        if (hm.data[k] < lo) lo = hm.data[k];
        if (hm.data[k] > hi) hi = hm.data[k];
    }
    for (int i = 0; i < n; i += downsample) {
        for (int j = 0; j < n; j += downsample) {
            float t = (hm(i,j) - lo) / (hi - lo);
            printf("%c", shading[(int)(t * 9)]);
        }
        printf("\n");
    }
}