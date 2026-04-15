#include "core/HeightMap.h"
#include <math.h>      // floorf
#include <stdlib.h>    // rand, srand

//-------------------------------------------------------
// Perlin Noise
//-------------------------------------------------------

// __constant__ on GPU = cached broadcast read-only memory
// Look up array with a random permutation of 0...255 (repeated twice for easy indexing)
__constant__ int d_P[512];\

// assures smooth changes between cells
__host__ __device__ static float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
// Interpolation function between (0,a)<->(1,b)
__host__ __device__ static float lerpf(float a, float b, float t) {
    return a + t * (b - a);
}
//
__host__ __device__ static float grad2(int hash, float x, float y) {
    switch (hash & 3) {
        case 0: return  x + y;
        case 1: return -x + y;
        case 2: return  x - y;
        default:return -x - y;
    }
}

// Pure cell logic — reads d_P
__device__
void perlin_cell(HeightMap hm, int i, int j,
                 int octaves, float persistence, float frequency)
        // i and j: cell indexes
        // octaves: the number of "layers"
        // persistence: change in amplitude between octaves (persistence*prev_amp) 
        // frequency: determines the periodicity of the gradient magnitude oscillation
        //          but it also has a random direction defined by the hash 
{
    float val = 0.0f, amp = 1.0f, freq = frequency, max_amp = 0.0f;
    // This loop repeats for each octave (doubles the frequency)
    for (int o = 0; o < octaves; o++) {
        // x and y are 
        float x = i * freq / hm.size;
        float y = j * freq / hm.size;

        int X = (int)floorf(x) & 255, Y = (int)floorf(y) & 255;

        x -= floorf(x); y -= floorf(y);
        float u = fade(x), v = fade(y);
        
        // Hashing using the random d_P
        int aa = d_P[d_P[X  ]+Y  ], ab = d_P[d_P[X  ]+Y+1],
            ba = d_P[d_P[X+1]+Y  ], bb = d_P[d_P[X+1]+Y+1];
        
        // Bilinear interpolation
        float noise = lerpf(lerpf(grad2(aa,x,     y    ), grad2(ba,x-1.0f,y    ), u),   //top horizontal
                            lerpf(grad2(ab,x,     y-1.0f), grad2(bb,x-1.0f,y-1.0f), u), // bottom
                            v); // vertical blend

        // Update values for the next octave
        val     += noise * amp;
        max_amp += amp;
        amp     *= persistence;
        freq    *= 2.0f;
    }
    hm(i, j) = val / max_amp;
}

__global__ // Classic one-task-many-times kernel 
void kernel_perlin(HeightMap hm, int octaves, float persistence, float frequency) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hm.size || j >= hm.size) return;
    perlin_cell(hm, i, j, octaves, persistence, frequency);
}

void terrain_perlin(HeightMap &hm, int octaves, float persistence,
                    float frequency, unsigned seed) {
    // Crea el random hash y lo manda a la GPU 
    srand(seed);
    int p[256];
    for (int i = 0; i < 256; i++) p[i] = i; // sequential array loading
    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1); 
        int tmp = p[i]; p[i] = p[j]; p[j] = tmp; // permutation i <-> j
    }
    int h_P[512];
    for (int i = 0; i < 512; i++) h_P[i] = p[i & 255]; // Doubles it for easy indexing
    CUDA_CHECK(cudaMemcpyToSymbol(d_P, h_P, 512 * sizeof(int)));

    // bloques, threads and kernel launch
    dim3 threads(16, 16);
    dim3 blocks((hm.size + 15) / 16, (hm.size + 15) / 16);

    kernel_perlin<<<blocks, threads>>>(hm, octaves, persistence, frequency);
    CUDA_CHECK(cudaDeviceSynchronize());
}

//-------------------------------------------------------
// Diamond-Square (host only — sequential by design) 
//                                        |-> Razon para no usarlo xd
//-------------------------------------------------------

static float randf(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

static void ds_diamond(HeightMap &hm, int x, int y, int half, float scale) {
    float avg = (hm(x,        y       ) +
                 hm(x,        y+2*half) +
                 hm(x+2*half, y       ) +
                 hm(x+2*half, y+2*half)) * 0.25f;
    hm(x+half, y+half) = avg + randf(-scale, scale);
}

static void ds_square(HeightMap &hm, int x, int y, int half, float scale) {
    int s = hm.size - 1;
    float sum = 0.0f; int cnt = 0;
    if (x-half >= 0) { sum += hm(x-half, y); cnt++; }
    if (x+half <= s) { sum += hm(x+half, y); cnt++; }
    if (y-half >= 0) { sum += hm(x, y-half); cnt++; }
    if (y+half <= s) { sum += hm(x, y+half); cnt++; }
    hm(x, y) = sum / cnt + randf(-scale, scale);
}

void terrain_diamond_square(HeightMap &hm, float initial_scale, float decay, unsigned seed) {
    srand(seed);
    int s = hm.size - 1;
    hm(0,0) = hm(0,s) = hm(s,0) = hm(s,s) = randf(-initial_scale, initial_scale);

    for (int step = s; step > 1; step /= 2) {
        int half = step / 2;
        float scale = initial_scale * powf(decay, log2f((float)s / step));
        for (int x = 0; x < s; x += step)
            for (int y = 0; y < s; y += step)
                ds_diamond(hm, x, y, half, scale);
        for (int x = 0; x <= s; x += half)
            for (int y = (x + half) % step; y <= s; y += step)
                ds_square(hm, x, y, half, scale);
    }
}