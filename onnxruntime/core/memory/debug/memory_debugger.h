#pragma once
#include <string>
#include <iostream>

namespace onnxruntime {
namespace memory {

class MemoryDebugger {
 public:
  static void DumpState(const std::string& tag);
  static void DumpAllocators();
};

}  // namespace memory
}  // namespace onnxruntime
