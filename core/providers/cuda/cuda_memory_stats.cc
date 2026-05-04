#include "cuda_memory_stats.h"
#include <cuda_runtime.h>
#include <sstream>

std::string CudaMemoryStats::ToString() const {
    std::ostringstream oss;
    oss << "CUDA Memory: Free=" << free_bytes
        << " total=" << total_bytes;
    return oss.str();
}
