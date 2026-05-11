#include "memory_profiler.h"

namespace onnxruntime {
namespace memory {

void MemoryProfiler::BeginRegion(const std::string& name) {
    std::cout << "[MemoryProfiler] Begin: " << name << std::endl;
}

void MemoryProfiler::EndRegion(const std::string& name) {
    std::cout << "[MemoryProfiler] End: " << name << std::endl;
}

}
}
