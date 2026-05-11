#pragma once
#include <string>

namespace onnxruntime {
namespace memory {

class MemoryPolicyRule {
 public:
  virtual ~MemoryPolicyRule() = default;
  virtual std::string Name() const { return "BaseRule"; }
  virtual bool Evaluate() const { return true; }
};

}
}
