#pragma once

namespace onnxruntime {
namespace memory {

class MemoryEpAdapter {
 public:
  virtual ~MemoryEpAdapter() = default;

  virtual void PublishMemorySnapshot() {}

  virtual const char* DomainName() const { return "unknown"; }
};

}
}
