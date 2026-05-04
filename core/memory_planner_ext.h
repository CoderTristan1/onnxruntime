#pragma once

#include <cstddef>

class MemoryPlannerExt {
public:
    virtual ~MemoryPlannerExt() = default;

    virtual void* AllocateActivation(std::size_t bytes) = 0;
    virtual void FreeActivation(void* ptr) = 0;

    virtual void BeginStep() = 0;
    virtual void EndStep() = 0;
};
