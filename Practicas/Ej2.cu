#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iomanip>

#include <cmath> // For fabsf
#include <cstdlib> // For rand(), srand(), RAND_MAX
#include <ctime>   // For time()

#define TILE_SIZE 16

// ---------------------------------------------------------
// 1. KERNEL FUSIONADO (Estudiante debe completar)
// ---------------------------------------------------------
__global__ void matmul_tanh_fused(float* A, float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // TODO: Calcular índices y lógica de Tiling + Tanh
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Indices del output C
    int col = bx * blockDim.x + tx;
    int row = by * blockDim.y + ty;

    if (row < N && col < N){

      // output element
      float Celem = 0;

      // Loop
      for (int i = 0; i < (N - 1) / TILE_SIZE + 1; i++) {
        // Cargo en la shared memory
        tile_A[ty][tx] = A[row*N + i*TILE_SIZE + tx];
        tile_B[ty][tx] = B[(i*TILE_SIZE + ty)*N + col];

        // Sincronizamos
        __syncthreads();

        // subLoop para calcular Celem
        for (int j = 0; j < TILE_SIZE; j++)
          Celem += tile_A[ty][j] * tile_B[j][tx];

        // Sincronizamos de nuevo
        __syncthreads();
      }

      // Cargamos el valor calculado dsp de aplicar la tanh (la f es xq es un float)
      C[row*N + col] = tanhf(Celem);
    }
}

// ---------------------------------------------------------
// 2. KERNEL ACTIVACIÓN (Estudiante debe completar)
// ---------------------------------------------------------
__global__ void apply_tanh(float* C, int size) {
    // TODO: Calcular índice y aplicar tanhf()
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = tanhf(C[idx]);
}

// ---------------------------------------------------------
// KERNEL COMPARACION
// ---------------------------------------------------------
bool compare_matrices_host(float* A, float* B, int size, float tol) {
    for (int i = 0; i < size; ++i) {
        if (fabsf(A[i] - B[i]) > tol) {
            return false;
        }
    }
    return true;
}

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t size = N * N * sizeof(float);

    float *h_A, *h_B, *h_C_cpu, *h_C_fused, *h_C_cublas;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C_cpu = (float*)malloc(size);
    h_C_fused = (float*)malloc(size);
    h_C_cublas = (float*)malloc(size);

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, size));
    checkCuda(cudaMalloc(&d_B, size));
    checkCuda(cudaMalloc(&d_C, size));

    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // --- CONFIGURACIÓN DE EVENTOS ---
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    float ms_fused = 0, ms_cublas = 0;

    // ---------------------------------------------------------
    // CPU Reference Calculation
    // ---------------------------------------------------------
    matmul_tanh_cpu(h_A, h_B, h_C_cpu, N);

    // ---------------------------------------------------------
    // MEDICIÓN TEST 1: KERNEL FUSIONADO
    // ---------------------------------------------------------
    // Reset d_C before fused kernel execution (esta de mas pero bueno)
    checkCuda(cudaMemset(d_C, 0, size));

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    checkCuda(cudaEventRecord(start)); // Marca inicio en el stream

    matmul_tanh_fused<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    checkCuda(cudaEventRecord(stop));  // Marca fin
    checkCuda(cudaEventSynchronize(stop)); // Espera a que la GPU termine
    checkCuda(cudaEventElapsedTime(&ms_fused, start, stop));
    checkCuda(cudaMemcpy(h_C_fused, d_C, size, cudaMemcpyDeviceToHost));

    // ---------------------------------------------------------
    // MEDICIÓN TEST 2: cuBLAS + ACTIVATION
    // ---------------------------------------------------------
    // Reset d_C before cuBLAS execution
    checkCuda(cudaMemset(d_C, 0, size)); // borra lo anterior (esta semi de mas)

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    checkCuda(cudaEventRecord(start));

    // TODO: Implementar cublasSgemm (C = A * B)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_A, N, d_B, N,
                &beta,
                d_C, N);

    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;
    apply_tanh<<<blocks, threads>>>(d_C, N * N);

    // Recomendacion del chat
    // cudaDeviceSynchronize();
      // Esto deja muy mal parado al cublas, no se si es realmente necesario
      // aumenta mucho el tiempo de ejecucion registrado

    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&ms_cublas, start, stop));
    checkCuda(cudaMemcpy(h_C_cublas, d_C, size, cudaMemcpyDeviceToHost));

    // --- VERIFICACIÓN DE RESULTADOS ---
    float tolerance = 1e-3; // Adjust tolerance as needed for floating-point comparisons
    bool fused_correct = compare_matrices_host(h_C_fused, h_C_cpu, N * N, tolerance);
    bool cublas_correct = compare_matrices_host(h_C_cublas, h_C_cpu, N * N, tolerance);


    // --- REPORTE DE RESULTADOS ---
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n========================================" << std::endl;
    std::cout << " Resultados para Matriz " << N << "x" << N << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "1. Tiempo Fused Kernel:      " << ms_fused << " ms" << std::endl;
    std::cout << "   Correctness:              " << (fused_correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "2. Tiempo cuBLAS + Tanh:     " << ms_cublas << " ms" << std::endl;
    std::cout << "   Correctness:              " << (cublas_correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    // Standard speedup calculation: Time of slower method / Time of faster method
    // Here, how many times faster Fused Kernel is compared to cuBLAS + Tanh
    std::cout << "Speedup (Fused / cuBLAS):    " << (ms_cublas/ms_fused) << "x" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // --- LIMPIEZA ---
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_fused); free(h_C_cublas);

    return 0;
}