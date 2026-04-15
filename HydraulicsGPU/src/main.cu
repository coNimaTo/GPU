#include "core/HeightMap.h"
#include "core/TerrainGen.h"
#include "utils/Plotter.h"

#define N 257

int main(int argc, char *argv[]) {
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

    hmap_print_ascii(hm, 8);
    hm.release();
    return 0;
}
