#include "cuda_training_stream.h"

CudaTrainingStream::CudaTrainingStream() {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
}

CudaTrainingStream::~CudaTrainingStream() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void CudaTrainingStream::Synchronize() {
    cudaStreamSynchronize(stream_);
}

void CudaTrainingStream::SetPriority(int priority) {
    int least, greatest;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);

    int clamped = std::max(greatest, std::min(priority, least));

    cudaStream_t new_stream;
    cudaStreamCreateWithPriority(&new_stream, cudaStreamNonBlocking, clamped);

    cudaStreamDestroy(stream_);
    stream_ = new_stream;
}
