#include "memory_tracking_handle.h"
#include "memory_tracker.h"

namespace onnxruntime {
namespace memory {

MemoryTrackingHandle::MemoryTrackingHandle(size_t size, const std::string& tag)
    : size_(size), tag_(tag), active_(true) {
  MemoryTracker::Instance().OnAlloc(size_, tag_);
}

MemoryTrackingHandle::~MemoryTrackingHandle() {
  if (active_) {
    MemoryTracker::Instance().OnFree(size_, tag_);
  }
}

MemoryTrackingHandle::MemoryTrackingHandle(MemoryTrackingHandle&& other) noexcept
    : size_(other.size_), tag_(other.tag_), active_(other.active_) {
  other.active_ = false;
}

MemoryTrackingHandle& MemoryTrackingHandle::operator=(MemoryTrackingHandle&& other) noexcept {
  if (this != &other) {
    if (active_) {
      MemoryTracker::Instance().OnFree(size_, tag_);
    }

    size_ = other.size_;
    tag_ = other.tag_;
    active_ = other.active_;

    other.active_ = false;
  }
  return *this;
}

}
}
