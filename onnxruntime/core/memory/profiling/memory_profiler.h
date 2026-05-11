#pragma once
#include <string>
#include <iostream>

namespace onnxruntime {
namespace memory {

class MemoryProfiler {
 public:
  void BeginRegion(const std::string& name);
  void EndRegion(const std::string& name);
};

}
}
