#pragma once
#include <string>

namespace onnxruntime {
namespace memory {

class MemoryEvents {
 public:
  static void Emit(const std::string& event_name);

  private:
   // TODO: hook into ORT
};

}
}
