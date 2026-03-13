# Ejercicio: Simulación de Difusión 1D

La temperatura $u$ de una barra unidimensional de longitud $L$ descrita por la **ecuación de difusión**:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

Utilizando un esquema de diferencias finitas hacia adelante en el tiempo y centradas en el espacio (**Euler explícito**), la actualización de la temperatura $u$ en la posición $i$ para el tiempo $n+1$ se puede discretizar como:

$$u_i^{n+1} = u_i^n + r(u_{i+1}^n - 2u_i^n + u_{i-1}^n)$$

donde $r = \alpha \frac{\Delta t}{\Delta x^2}$ es el parámetro de control. Para que el método sea numéricamente estable, se debe cumplir la condición de Courant: $r \leq 0.5$.

1. Completar el template provisto del programa en **CUDA C** que evolucione un **pulso cuadrado** de temperatura en el centro de la barra y registre la evolución del punto central $u(L/2)$ a lo largo del tiempo, que cumple las siguientes condiciones,

  * **Condición Inicial:** Definir un perfil donde $u = 1.0$ en la región central (por ejemplo, $L/4 < i < 3L/4$) y $u = 0.0$ en el resto de la barra.
  * **Condiciones de Contorno:** Aplicar condiciones de **Dirichlet absorbentes**, manteniendo los extremos fijos: $u_0 = u_{L-1} = 0$.
  * **Gestión de Memoria (Double Buffering):**
    * Utilizar dos arreglos en la GPU (`d_u_old` y `d_u_new`) para evitar la lectura de valores que ya han sido modificados por otros hilos en el mismo paso temporal (*race condition*).
    * Realizar el intercambio (*swap*) de punteros en el Host (CPU) al finalizar cada iteración del bucle temporal.
  * **Optimización de Transferencia (Bus PCIe):**
    * **Crucial:** No copiar el vector completo de vuelta al Host en cada paso de tiempo.
    * Dado que solo nos interesa la evolución del centro, realizar una transferencia selectiva de **un único valor** `float` desde la dirección de memoria `&d_u_old[L/2]` hacia una variable local en el Host mediante `cudaMemcpy`.

2. Graficar el valor central de la temperatura de la barra en función del tiempo.

3. Graficar el tiempo de ejecución vs $L=1024, 2048, \dots, 4194304$.

4. *Opcional*: hacer una pelicula con todo el perfil $T(x,t)$ vs $t$.