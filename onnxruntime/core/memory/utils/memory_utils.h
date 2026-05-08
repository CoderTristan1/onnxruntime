#pragma once
namespace onnxruntime {
namespace memory {

class MemoryUtils {
 public:
  static size_t Align(size_t value, size_t alignment);
};

}
}
