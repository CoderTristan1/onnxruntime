#pragma once
#include <string>

namespace onnxruntime {
namespace memory {

class MemoryConfig {
 public:
  static std::string GetDefaultConfig();
};

}
}
