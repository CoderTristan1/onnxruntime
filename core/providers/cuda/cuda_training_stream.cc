#include "cuda_training_stream.h"

CudaTrainingStream::CudaTrainingStream() {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
}

CudaTrainingStream::~CudaTrainingStream() {
    if (stream_) cudaStreamDestroy(stream_);
}

void CudaTrainingStream::Synchronize() {
    cudaStreamSynchronize(stream_);
}
