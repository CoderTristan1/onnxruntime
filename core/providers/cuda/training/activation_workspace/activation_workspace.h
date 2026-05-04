#pragma once

#include <cstddef>
#include "activation_suballocator.h"

struct ActivationWorkspaceStats {
    std::size_t current_workspace_size_bytes;
    std::size_t peak_usage_bytes;
    std::size_t expansion_count;
    std::size_t allocations_this_step;
};

class ActivationWorkspace {
public:
    bool Initialize(std::size_t initial_bytes, double max_fraction);

    void* Allocate(std::size_t bytes);
    void  Free(void* ptr);

    void BeginStep();
    void EndStep();

    ActivationWorkspaceStats GetStats() const;

private:
    ActivationSuballocator suballocator_;
    std::size_t allocations_this_step_ = 0;
};
