#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include "core/memory/memory_info.h"

namespace onnxruntime {
namespace memory {

struct MemorySnapshot {
    MemoryInfo info;
    std::uint64_t bytes_used{0};
    std::uint64_t bytes_reserved{0};
};

class MemoryDiagnostics {
 public:
  using SnapshotCallback = std::function<void(const MemorySnapshot&)>;

  void RegisterSnapshotCallback(SnapshotCallback cb);
  void PublishSnapshot(const MemorySnapshot& snapshot);

  private:
   SnapshotCallback callback_;

};

}
}
