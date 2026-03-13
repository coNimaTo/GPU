#include <iostream>
#include <vector>
#include <cstdio>
#include <cuda_runtime.h>

// Kernel que implementa el stencil de diferencias finitas 1D
__global__ void diffusionKernel(float *u_old, float *u_new, float r, int L) {
  // COMPLETAR
}

int main() {
    // 1. Configuración de parámetros
    const int L = 512;            // Puntos en la barra
    const int steps = 20000;      // Pasos temporales
    const int output_freq = 100;  // Frecuencia de muestreo del centro
    const float r = 0.4f;         // Coeficiente de estabilidad (r <= 0.5)

    size_t size = L * sizeof(float);

    // 2. Inicialización del perfil (Pulso Cuadrado)
    std::vector<float> h_u(L, 0.0f);
    for (int i = L / 4; i < 3 * L / 4; ++i) {
        h_u[i] = 1.0f;
    }

    // 3. Reserva de memoria en GPU y Double Buffering
    float *d_u_old, *d_u_new;
    cudaMalloc(&d_u_old, size);
    cudaMalloc(&d_u_new, size);

    // Copiar condición inicial y limpiar el segundo buffer
    cudaMemcpy(d_u_old, h_u.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_u_new, 0, size);

    // 4. Parámetros de ejecución de CUDA
    int threadsPerBlock = 256;
    int blocksPerGrid = (L + threadsPerBlock - 1) / threadsPerBlock;

    std::printf("# Paso \t Temperatura_Centro\n");


    // Agregar timers de GPU para medir el tiempo de ejecución de todo el loop
    // COMPLETAR

    // 5. Bucle de evolución temporal
    for (int s = 0; s <= steps; ++s) {

        // --- OPTIMIZACIÓN DE TRANSFERENCIA ---
        // Extraemos solo el punto central del buffer actual antes del swap
        if (s % output_freq == 0) {
            // Se transfiere un único float (4 bytes) evitando copiar todo el array
            // COMPLETAR

            // Se imprime ese único float en el host
            // COMPLETAR
        }

        // Lanzamiento del kernel
        diffusionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_u_old, d_u_new, r, L);

        // --- GESTIÓN DE MEMORIA (Swap de punteros) ---
        // Intercambiamos los buffers: el "nuevo" pasa a ser el "viejo" para el siguiente paso
        float *temp = d_u_old;
        d_u_old = d_u_new;
        d_u_new = temp;
    }

    // Imprimir el tiempo de ejecución de todo el loop
    // COMPLETAR

    // 6. Liberación de memoria
    cudaFree(d_u_old);
    cudaFree(d_u_new);

    return 0;
}