#include "provider_registry.h"

ProviderRegistry& ProviderRegistry::Instance() {
    static ProviderRegistry instance;
    return instance;
}

void ProviderRegistry::Register(const std::string& name, void* factory) {
    factories_[name] = factory;
}

void* ProviderRegistry::GetFactory(const std::string& name) const {
    auto it = factories_.find(name);
    return it == factories_.end() ? nullptr : it->second;
}
