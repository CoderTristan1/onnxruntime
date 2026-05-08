#pragma once
#include <string>

namespace onnxruntime {
namespace memory {

class MemoryTelemetry {
 public:
  static void Log(const std::string& msg);
};

}
}
