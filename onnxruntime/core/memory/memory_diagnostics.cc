#include "core/memory/memory_diagnostics.h"

namespace onnxruntime {
namespace memory {

void MemoryDiagnostics::RegisterSnapshotCallback(SnapshotCallback cb) {
  callback_ = std::move(cb);
}

void MemoryDiagnostics::PublishSnapshot(const MemorySnapshot& snapshot) {
  if (callback_) {
    callback_(snapshot);
  }
}

}  // namespace memory
}  // namespace onnxruntime
