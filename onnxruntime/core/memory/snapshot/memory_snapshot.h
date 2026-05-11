#pragma once
#include <cstdint>
#include <vector>

namespace onnxruntime {
namespace memory {

struct MemorySnapshotEntry {
  uint64_t address;
  size_t size;
};

class MemorySnapshot {
 public:
  void Capture();
  const std::vector<MemorySnapshotEntry>& Entries() const;

 private:
  std::vector<MemorySnapshotEntry> entries_;
};

}
}
