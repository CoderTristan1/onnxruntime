#include "memory_debugger.h"

namespace onnxruntime {
namespace memory {

void MemoryDebugger::DumpState(const std::string& tag) {
  std::cout << "[MemoryDebugger] DumpState: " << tag << std::endl;
}

void MemoryDebugger::DumpAllocators() {
  std::cout << "[MemoryDebugger] DumpAllocators: no allocators registered" << std::endl;
}

}
}
