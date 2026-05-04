#include "cuda_provider_debug.h"
#include <cuda_runtime.h>
#include <sstream>

std::string CudaProviderDebug::DumpDeviceInfo() {
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::ostringstream oss;
    oss << "CUDA Device Info:\n"
        << "  Name: " << prop.name << "\n"
        << "  SMs: " << prop.multiProcessorCount << "\n"
        << "  Global Mem: " << prop.totalGlobalMem << "\n"
        << "  Shared Mem per Block: " << prop.sharedMemPerBlock << "\n"
        << "  Warp Size: " << prop.warpSize << "\n";

    return oss.str();
}
