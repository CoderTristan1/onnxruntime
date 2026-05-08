#include "memory_policy_manager.h"

namespace onnxruntime {
namespace memory {

void MemoryPolicyManager::SetPolicy(Policy p) {
    current_policy_ = p;
}

MemoryPolicyManager::Policy MemoryPolicyManager::GetPolicy() const {
    return current_policy_;
}

}
}
