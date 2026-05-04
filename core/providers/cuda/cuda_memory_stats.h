#pragma once

#include <cstddef>
#include <string>

struct CudaMemoryStats {
    std::size_t free_bytes;
    std::size_t total_bytes;

    std::string ToString() const;
};
