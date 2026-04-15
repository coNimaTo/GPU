#include <iostream>
#include <vector>
#include <cstdio>
#include <cuda_runtime.h>
#include "gpu_timer.h"

// Kernel que implementa el stencil de diferencias finitas 1D
__global__ void diffusionKernel(float *u_old, float *u_new, float r, int L) {
    // indice de thread mapeado a indice de array
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Checkea si esta dentro de los limites
        // Dejamos fuera los extremos, que estan fijos
    if (i >= 1 && i < L - 1)
    {
        u_new[i] = u_old[i] + r * (u_old[i+1] - 2.0f * u_old[i] + u_old[i-1]);
    }
}

int main(int argc, char **argv) {
    // 1. Configuración de parámetros
    int L = 512;            // Puntos en la barra
    int steps = 20000;      // Pasos temporales
    const int output_freq = 100;  // Frecuencia de muestreo del centro
    const float r = 0.4f;         // Coeficiente de estabilidad (r <= 0.5)

    if(argc > 1) L=atoi(argv[1]);
    if(argc > 2) steps=atoi(argv[2]);

    // Check for enough arguments: L, steps, and temperature data filename
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <L> <steps> <temperature_data_filename>\n", argv[0]);
        return 1;
    }
    const char* temp_data_filename = argv[3]; // Filename for temperature data (e.g., T_L.dat)
    const char* time_data_filename = "time_data.dat"; // Fixed filename for time data

    // Open file for temperature data
    FILE* temp_data_file = fopen(temp_data_filename, "w"); // 'w' to overwrite/create
    if (temp_data_file == NULL) {
        fprintf(stderr, "Error: Could not open temperature data file %s for writing.\n", temp_data_filename);
        return 1;
    }

    // Open file for execution times
    FILE* time_data_file = fopen(time_data_filename, "a"); // 'a' to append
    if (time_data_file == NULL) {
        fprintf(stderr, "Error: Could not open time data file %s for writing.\n", time_data_filename);
        fclose(temp_data_file); // Close already opened file before exiting
        return 1;
    }

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

    // Write header to temperature data file
    fprintf(temp_data_file, "# Paso \t Temperatura_Centro\n");

    // Agregar timers de GPU para medir el tiempo de ejecución de todo el loop
    gpu_timer Reloj;
	  Reloj.tic();

    // variable donde voy a guardar el dato que extraiga del centro
    float Temp_Centro;

    // 5. Bucle de evolución temporal
    for (int s = 0; s <= steps; ++s) {

        // --- OPTIMIZACIÓN DE TRANSFERENCIA ---
        // Extraemos solo el punto central del buffer actual antes del swap
        if (s % output_freq == 0) {
            // Se transfiere un único float (4 bytes) evitando copiar todo el array
            cudaMemcpy( &Temp_Centro, d_u_old + (L/2), sizeof(float), cudaMemcpyDeviceToHost );

            // Se imprime ese único float en el temperatura data file
            fprintf(temp_data_file, "%d\t%f\n", s, Temp_Centro);
        }

        // Lanzamiento del kernel
        diffusionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_u_old, d_u_new, r, L);

        // --- GESTIÓN DE MEMORIA (Swap de punteros) ---
        // Intercambiamos los buffers: el "nuevo" pasa a ser el "viejo" para el siguiente paso
        float *temp = d_u_old;
        d_u_old = d_u_new;
        d_u_new = temp;
    }

    // Imprimir el tiempo de ejecución de todo el loop al archivo de tiempos
    fprintf(time_data_file, "%d\t%d\t%lf\n", L, steps, Reloj.tac());

    // 6. Liberación de memoria y cierre de archivos
    fclose(temp_data_file);
    fclose(time_data_file);

    cudaFree(d_u_old);
    cudaFree(d_u_new);

    return 0;
}