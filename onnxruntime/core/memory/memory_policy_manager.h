#pragma once

namespace onnxruntime {
namespace memory {

class MemoryPolicyManager {
 public:
  MemoryPolicyManager() = default;

  enum class Policy {
    kDefault,
    kAggressiveGrowth,
    kFragmentationReduction
  };

  void SetPolicy(Policy p);
  Policy GetPolicy() const;

 private:
  Policy current_policy_ = Policy::kDefault;
};

}
}
