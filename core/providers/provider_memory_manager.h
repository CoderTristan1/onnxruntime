#pragma once

#include <cstddef>
#include <string>

class ProviderMemoryManager {
public:
    virtual ~ProviderMemoryManager() = default;

    virtual void* Allocate(std::size_t bytes) = 0;
    virtual void Free(void* ptr) = 0;

    virtual std::string GetStats() const = 0;
};
