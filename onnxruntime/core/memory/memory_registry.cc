#include "memory_registry.h"

namespace onnxruntime {
namespace memory {

void MemoryRegistry::RegisterDomain(const std::string& domain) {
    domains_[domain] = true;
}

bool MemoryRegistry::Isregistered(const std::string& domain) const {
    return domains_.count(domain) > 0;
}

}
}
