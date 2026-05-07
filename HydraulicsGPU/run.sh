#! /bin/bash

# El sistema de cola devuelve una variable $SGE_GPU que contiene los IDs de los dispositivos requeridos (separados por coma). 
# Ejemplo: 0 o 0,1 dependiendo del numero de recursos pedidos
# Use este device ID para cudaSetDevice()

#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpushort
#$ -l gpu=1
#$ -l memoria_a_usar=1G
#$ -N TestRun
#
# cargar variables de entorno para encontrar cuda

nvidia-smi
module load cuda/12.2.1-gcc-11.1.0-fvljoe5

echo DeviceID: $SGE_GPU

#ejecutar binario con sus respectivos argumentos
./terrain 4
#compute-sanitizer --tool memcheck ./terrain
