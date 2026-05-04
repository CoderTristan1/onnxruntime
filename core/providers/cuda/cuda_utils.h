#pragma once

#include <cuda_runtime.h>
#include <string>

inline const char* CudaErrToStr(cudaError_t err) {
    return cudaGetErrorString(err);
}

inline void CudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s)\n", msg, cudaGetErrorString(err));
    }
}
