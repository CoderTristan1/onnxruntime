#pragma once

#include <cuda_runtime.h>

class CudaTrainingStream {
public:
    CudaTrainingStream();
    ~CudaTrainingStream();

    cudaStream_t Stream() const { return stream_; }
    void Synchronize();

    private:
        cudaStream_t stream_ = nullptr;
};
