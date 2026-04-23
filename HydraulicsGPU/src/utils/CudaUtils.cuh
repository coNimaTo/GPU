#pragma once
#include <stdio.h>     // printf
#include <stdlib.h>    // calloc, free
#include <string.h>    // memset, memcpy

#ifndef __CUDACC__
  #define __host__
  #define __device__
  #define __global__
  #define __constant__
  #include <stdlib.h>
  #include <string.h>
  inline void* cudaMallocManaged_stub(size_t n) { return calloc(n, 1); }
  #define cudaMallocManaged(ptr, size) \
      (*(ptr) = (__typeof__(**(ptr))*)cudaMallocManaged_stub(size), 0)
  #define cudaFree(ptr)               (free(ptr), 0)
  #define cudaMemset(p,v,n)           (memset(p,v,n), 0)
  #define cudaDeviceSynchronize()     0
  #define cudaMemcpyToSymbol(d,s,n)   (memcpy(d,s,n), 0)
  #define CUDA_CHECK(call)            (call)
  struct dim3 { int x,y,z; dim3(int x=1,int y=1,int z=1):x(x),y(y),z(z){} };
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