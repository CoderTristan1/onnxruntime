#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <mutex>

namespace onnxruntime {
namespace memory {

struct MemoryTrackerStats {
  uint64_t total_bytes = 0;
  uint64_t peak_bytes = 0;
  uint64_t alloc_count = 0;
  uint64_t free_count = 0;
};

struct MemoryTagStats {
  uint64_t bytes = 0;
  uint64_t alloc_count = 0;
  uint64_t free_count = 0;
};

class MemoryTracker {
 public:
  static MemoryTracker& Instance();

  void OnAlloc(size_t size, const std::string& tag);
  void OnFree(size_t size, const std::string& tag);

  MemoryTrackerStats GetGlobalStats() const;
  std::unordered_map<std::string, MemoryTagStats> GetTagStats() const;

  void Reset();

 private:
  MemoryTracker() = default;

  mutable std::mutex mutex_;
  MemoryTrackerStats global_;
  std::unordered_map<std::string, MemoryTagStats> per_tag_;
};

}
}
