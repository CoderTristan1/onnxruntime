#pragma once

#include <cstddef>
#include <string>

struct TrainingStateSnapshot {
    size_t step_number;
    size_t peak_activation_memory;
    size_t workspace_expansions;

    std::string ToString() const;
};
