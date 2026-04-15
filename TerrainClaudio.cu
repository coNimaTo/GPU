// Terrain.cu
// Compile with CUDA:  nvcc -O2 -arch=sm_75 Terrain.cu -o terrain
// Compile for CPU:    g++ -O2 -std=c++11 -x c++ Terrain.cu -o terrain -lm

#ifndef __CUDACC__
  // Stubs so the file compiles with g++ without any changes
  #define __host__
  #define __device__
  #define __global__
  #define __constant__
  #include <stdlib.h>
  #include <string.h>
  inline void* cudaMallocManaged_stub(size_t n) { return calloc(n, 1); }
  #define cudaMallocManaged(ptr, size) (*(ptr) = (__typeof__(**(ptr))*)cudaMallocManaged_stub(size), 0)
  #define cudaFree(ptr)               (free(ptr), 0)
  #define cudaMemset(p,v,n)           (memset(p,v,n), 0)
  #define cudaDeviceSynchronize()     0
  #define cudaMemcpyToSymbol(dst, src, size) (memcpy(dst, src, size), 0)
  #define CUDA_CHECK(call)            (call)
  struct dim3 { int x, y, z; dim3(int x=1,int y=1,int z=1):x(x),y(y),z(z){} };
#else
  #include <cuda_runtime.h>
  #define CUDA_CHECK(call)                                                \
    do {                                                                  \
      cudaError_t err = (call);                                           \
      if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d - %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(err));             \
        exit(1);                                                          \
      }                                                                   \
    } while(0)
#endif

#include <stdio.h>
#include <math.h>
#include <time.h>

//-------------------------------------------------------
// CPU kernel launcher
// Simulates a 2D thread grid by calling functor f(i, j)
// for every cell. Functor F: void(int i, int j)
//-------------------------------------------------------

#ifndef __CUDACC__
template<typename F>
void cpu_launch(dim3 blocks, dim3 threads, F f, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            f(i, j);
}
#endif

//-------------------------------------------------------
// HeightMap
//-------------------------------------------------------

#define N 257   // must be 2^k + 1 for Diamond-Square

struct HeightMap {
    float *data;
    int    size;

    __host__ __device__
    float& operator()(int row, int col) {
        return data[row * size + col];
    }

    __host__ __device__
    const float& operator()(int row, int col) const {
        return data[row * size + col];
    }

    void allocate(int n) {
        size = n;
        CUDA_CHECK(cudaMallocManaged(&data, (size_t)n * n * sizeof(float)));
        CUDA_CHECK(cudaMemset(data, 0, (size_t)n * n * sizeof(float)));
    }

    void release() {
        CUDA_CHECK(cudaFree(data));
        data = nullptr;
    }
};

//-------------------------------------------------------
// Fault Lines
//-------------------------------------------------------

// Pure cell logic — __host__ __device__ so it compiles for both
__host__ __device__
void fault_lines_cell(HeightMap hm, int i, int j,
                      const float *fx1, const float *fy1,
                      const float *fx2, const float *fy2,
                      const float *fdelta, int n_faults)
{
    float val = 0.0f;
    for (int f = 0; f < n_faults; f++) {
        float side = (fx2[f] - fx1[f]) * (i - fy1[f])
                   - (fy2[f] - fy1[f]) * (j - fx1[f]);
        val += (side > 0.0f) ? fdelta[f] : -fdelta[f];
    }
    hm(i, j) = val;
}

#ifdef __CUDACC__
__global__
void kernel_fault_lines(HeightMap hm,
                        const float *fx1, const float *fy1,
                        const float *fx2, const float *fy2,
                        const float *fdelta, int n_faults)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hm.size || j >= hm.size) return;
    fault_lines_cell(hm, i, j, fx1, fy1, fx2, fy2, fdelta, n_faults);
}
#endif

void terrain_fault_lines(HeightMap &hm, int n_faults, float delta_max) {
    srand((unsigned)time(NULL));

    // Use Unified Memory for fault params — works on both CPU and GPU
    float *fx1, *fy1, *fx2, *fy2, *fdelta;
    size_t fsz = n_faults * sizeof(float);
    CUDA_CHECK(cudaMallocManaged(&fx1,    fsz));
    CUDA_CHECK(cudaMallocManaged(&fy1,    fsz));
    CUDA_CHECK(cudaMallocManaged(&fx2,    fsz));
    CUDA_CHECK(cudaMallocManaged(&fy2,    fsz));
    CUDA_CHECK(cudaMallocManaged(&fdelta, fsz));

    for (int f = 0; f < n_faults; f++) {
        fx1[f]    = (float)rand() / RAND_MAX * hm.size;
        fy1[f]    = (float)rand() / RAND_MAX * hm.size;
        fx2[f]    = (float)rand() / RAND_MAX * hm.size;
        fy2[f]    = (float)rand() / RAND_MAX * hm.size;
        fdelta[f] = delta_max * (1.0f - (float)f / n_faults);
    }

    dim3 threads(16, 16);
    dim3 blocks((hm.size + 15) / 16, (hm.size + 15) / 16);

#ifdef __CUDACC__
    kernel_fault_lines<<<blocks, threads>>>(hm, fx1, fy1, fx2, fy2, fdelta, n_faults);
    CUDA_CHECK(cudaDeviceSynchronize());
#else
    auto fn = [hm, fx1, fy1, fx2, fy2, fdelta, n_faults](int i, int j) mutable {
        fault_lines_cell(hm, i, j, fx1, fy1, fx2, fy2, fdelta, n_faults);
    };
    cpu_launch(blocks, threads, fn, hm.size);
#endif

    CUDA_CHECK(cudaFree(fx1)); CUDA_CHECK(cudaFree(fy1));
    CUDA_CHECK(cudaFree(fx2)); CUDA_CHECK(cudaFree(fy2));
    CUDA_CHECK(cudaFree(fdelta));
}

//-------------------------------------------------------
// Perlin Noise
//-------------------------------------------------------

// __constant__ on GPU = cached broadcast read-only memory
// On CPU it's just a static array — same access syntax either way
#ifdef __CUDACC__
  __constant__ int d_P[512];
#else
  static int d_P[512];
#endif

__host__ __device__ static float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
__host__ __device__ static float lerpf(float a, float b, float t) {
    return a + t * (b - a);
}
__host__ __device__ static float grad2(int hash, float x, float y) {
    switch (hash & 3) {
        case 0: return  x + y;
        case 1: return -x + y;
        case 2: return  x - y;
        default:return -x - y;
    }
}

// Pure cell logic — reads d_P which resolves correctly on both CPU/GPU
__host__ __device__
void perlin_cell(HeightMap hm, int i, int j,
                 int octaves, float persistence, float frequency)
{
    float val = 0.0f, amp = 1.0f, freq = frequency, max_amp = 0.0f;
    for (int o = 0; o < octaves; o++) {
        float x = i * freq / hm.size;
        float y = j * freq / hm.size;
        int X = (int)floorf(x) & 255, Y = (int)floorf(y) & 255;
        x -= floorf(x); y -= floorf(y);
        float u = fade(x), v = fade(y);
        int aa = d_P[d_P[X  ]+Y  ], ab = d_P[d_P[X  ]+Y+1],
            ba = d_P[d_P[X+1]+Y  ], bb = d_P[d_P[X+1]+Y+1];
        float noise = lerpf(lerpf(grad2(aa,x,     y    ), grad2(ba,x-1.0f,y    ), u),
                            lerpf(grad2(ab,x,     y-1.0f), grad2(bb,x-1.0f,y-1.0f), u), v);
        val     += noise * amp;
        max_amp += amp;
        amp     *= persistence;
        freq    *= 2.0f;
    }
    hm(i, j) = val / max_amp;
}

#ifdef __CUDACC__
__global__
void kernel_perlin(HeightMap hm, int octaves, float persistence, float frequency) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hm.size || j >= hm.size) return;
    perlin_cell(hm, i, j, octaves, persistence, frequency);
}
#endif

void terrain_perlin(HeightMap &hm, int octaves, float persistence,
                    float frequency, unsigned seed) {
    srand(seed);
    int p[256];
    for (int i = 0; i < 256; i++) p[i] = i;
    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = p[i]; p[i] = p[j]; p[j] = tmp;
    }
    int h_P[512];
    for (int i = 0; i < 512; i++) h_P[i] = p[i & 255];
    CUDA_CHECK(cudaMemcpyToSymbol(d_P, h_P, 512 * sizeof(int)));

    dim3 threads(16, 16);
    dim3 blocks((hm.size + 15) / 16, (hm.size + 15) / 16);

#ifdef __CUDACC__
    kernel_perlin<<<blocks, threads>>>(hm, octaves, persistence, frequency);
    CUDA_CHECK(cudaDeviceSynchronize());
#else
    auto fn = [hm, octaves, persistence, frequency](int i, int j) mutable {
        perlin_cell(hm, i, j, octaves, persistence, frequency);
    };
    cpu_launch(blocks, threads, fn, hm.size);
#endif
}

//-------------------------------------------------------
// Diamond-Square (host only — sequential by design)
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

void terrain_diamond_square(HeightMap &hm, float initial_scale, float decay) {
    srand((unsigned)time(NULL));
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

//-------------------------------------------------------
// Main
//-------------------------------------------------------

int main(int argc, char *argv[]) {
    HeightMap hm;
    hm.allocate(N);

    int algorithm = 0;
    if (argc > 1) {
        algorithm = atoi(argv[1]);
    }

    switch (algorithm) {
        case 1:
            terrain_diamond_square(hm, 1.0f, 0.6f);
            break;
        case 2:
            terrain_perlin(hm, 6, 0.5f, 4.0f, 42);
            break;
        case 3:
            terrain_fault_lines(hm, 200, 1.0f);
            break;
        default:
            terrain_perlin(hm, 6, 0.5f, 4.0f, 42);
            break;
    }

    hmap_print_ascii(hm, 8);
    hm.release();
    return 0;
}
