#include "memory_tracker.h"

namespace onnxruntime {
namespace memory {

MemoryTracker& MemoryTracker::Instance() {
  static MemoryTracker instance;
  return instance;
}

void MemoryTracker::OnAlloc(size_t size, const std::string& tag) {
  std::lock_guard<std::mutex> lock(mutex_);

  global_.total_bytes += size;
  global_.alloc_count += 1;
  if (global_.total_bytes > global_.peak_bytes) {
    global_.peak_bytes = global_.total_bytes;
  }

  auto& t = per_tag_[tag];
  t.bytes += size;
  t.alloc_count += 1;
}

void MemoryTracker::OnFree(size_t size, const std::string& tag) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (global_.total_bytes >= size) {
    global_.total_bytes -= size;
  } else {
    global_.total_bytes = 0;
  }
  global_.free_count += 1;

  auto it = per_tag_.find(tag);
  if (it != per_tag_.end()) {
    auto& t = it->second;
    if (t.bytes >= size) {
      t.bytes -= size;
    } else {
      t.bytes = 0;
    }
    t.free_count += 1;
  }
}

MemoryTrackerStats MemoryTracker::GetGlobalStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return global_;
}

std::unordered_map<std::string, MemoryTagStats> MemoryTracker::GetTagStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return per_tag_;
}

void MemoryTracker::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  global_ = MemoryTrackerStats{};
  per_tag_.clear();
}

}  // namespace memory
}  // namespace onnxruntime
