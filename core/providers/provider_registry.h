#pragma once

#include <string>
#include <unordered_map>

class ProviderRegistry {
public:
    static ProviderRegistry& Instance();

    void Register(const std::string& name, void* factory);
    void* GetFactory(const std::string& name) const;

private:
    ProviderRegistry() = default;

    std::unordered_map<std::string, void*> factories_;
};
