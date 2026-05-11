#include "memory_snapshot.h"

namespace onnxruntime {
namespace memory {

void MemorySnapshot::Capture() {
  entries_.clear();
  entries_.push_back({0xDEADBEEF, 4096});
}

const std::vector<MemorySnapshotEntry>& MemorySnapshot::Entries() const {
  return entries_;
}

}
}
