#pragma once

#include <cstddef>
#include <string>

namespace onnxruntime {
namespace memory {

class MemoryTrackingHandle {
 public:
  MemoryTrackingHandle(size_t size, const std::string& tag);
  ~MemoryTrackingHandle();

  MemoryTrackingHandle(const MemoryTrackingHandle&) = delete;
  MemoryTrackingHandle& operator=(const MemoryTrackingHandle&) = delete;

  MemoryTrackingHandle(MemoryTrackingHandle&& other) noexcept;
  MemoryTrackingHandle& operator=(MemoryTrackingHandle&& other) noexcept;

  size_t Size() const { return size_; }
  const std::string& Tag() const { return tag_; }

private:
 size_t size_;
 std::string tag_;
 bool active_;
};

}
}
