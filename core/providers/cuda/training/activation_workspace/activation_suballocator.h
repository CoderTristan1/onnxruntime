#pragma once

#include <cstddef>
#include <vector>

struct ActivationBlock {
    std::size_t offset;
    std::size_t size;
    bool free;
};

class ActivationSuballocator {
public:
    ActivationSuballocator();
    ~ActivationSuballocator();

    bool Initialize(std::size_t initial_bytes, double max_fraction);

    void* Allocate(std::size_t bytes);
    void  Free(void* ptr);

    void  ResetStep();

    std::size_t Capacity() const { return capacity_bytes_; }
    std::size_t PeakUsage() const { return peak_usage_bytes_; }
    std::size_t ExpansionCount() const { return expansion_count_; }

private:
    bool Grow(std::size_t min_extra_bytes);

    char* base_ptr_ = nullptr;              // GPU memory
    std::size_t capacity_bytes_ = 0;

    std::vector<ActivationBlock> blocks_;
    std::size_t current_usage_bytes_ = 0;
    std::size_t peak_usage_bytes_ = 0;
    std::size_t expansion_count_ = 0;

    double max_fraction_ = 0.8;
};
