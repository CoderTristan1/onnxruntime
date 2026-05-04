#include "activation_suballocator.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace {
static std::size_t AlignUp(std::size_t x, std::size_t align) {
    return (x + align - 1) & ~(align - 1);
}
}

ActivationSuballocator::ActivationSuballocator() = default;

ActivationSuballocator::~ActivationSuballocator() {
    if (base_ptr_) {
        cudaFree(base_ptr_);
        base_ptr_ = nullptr;
    }
}

bool ActivationSuballocator::Initialize(std::size_t initial_bytes, double max_fraction) {
    max_fraction_ = max_fraction;

    if (base_ptr_ != nullptr)
        return true;

    if (cudaMalloc(&base_ptr_, initial_bytes) != cudaSuccess)
        return false;

    capacity_bytes_ = initial_bytes;
    blocks_.clear();
    blocks_.push_back({0, capacity_bytes_, true});
    current_usage_bytes_ = 0;
    peak_usage_bytes_ = 0;
    expansion_count_ = 0;

    return true;
}

void* ActivationSuballocator::Allocate(std::size_t bytes) {
    if (!base_ptr_)
        return nullptr;

    const std::size_t aligned = AlignUp(bytes, 256);

    for (auto& b : blocks_) {
        if (b.free && b.size >= aligned) {

            if (b.size > aligned) {
                ActivationBlock new_block{
                    b.offset + aligned,
                    b.size - aligned,
                    true
                };
                b.size = aligned;
                blocks_.insert(std::next(blocks_.begin(), &b - &blocks_[0]), new_block);
            }

            b.free = false;
            current_usage_bytes_ += b.size;
            peak_usage_bytes_ = std::max(peak_usage_bytes_, current_usage_bytes_);
            return base_ptr_ + b.offset;
        }
    }

    if (!Grow(aligned))
        return nullptr;

    return Allocate(bytes);
}

void ActivationSuballocator::Free(void* ptr) {
    if (!ptr || !base_ptr_)
        return;

    std::size_t offset = static_cast<char*>(ptr) - base_ptr_;

    for (size_t i = 0; i < blocks_.size(); ++i) {
        auto& b = blocks_[i];
        if (b.offset == offset && !b.free) {
            b.free = true;
            current_usage_bytes_ -= b.size;

            for (size_t j = 1; j < blocks_.size(); ++j) {
                if (blocks_[j].free && blocks_[j - 1].free &&
                    blocks_[j - 1].offset + blocks_[j - 1].size == blocks_[j].offset) {
                    blocks_[j - 1].size += blocks_[j].size;
                    blocks_.erase(blocks_.begin() + j);
                    --j;
                }
            }
            return;
        }
    }
}

void ActivationSuballocator::ResetStep() {
    for (auto& b : blocks_)
        b.free = true;

    current_usage_bytes_ = 0;
}

bool ActivationSuballocator::Grow(std::size_t min_extra_bytes) {
    std::size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    std::size_t max_allowed = static_cast<std::size_t>(total_bytes * max_fraction_);
    std::size_t new_capacity = capacity_bytes_;

    while (new_capacity < capacity_bytes_ + min_extra_bytes)
        new_capacity *= 2;

    if (new_capacity > max_allowed)
        return false;

    char* new_base = nullptr;
    if (cudaMalloc(&new_base, new_capacity) != cudaSuccess)
        return false;

    cudaMemcpy(new_base, base_ptr_, capacity_bytes_, cudaMemcpyDeviceToDevice);
    cudaFree(base_ptr_);

    base_ptr_ = new_base;
    blocks_.push_back({capacity_bytes_, new_capacity - capacity_bytes_, true});
    capacity_bytes_ = new_capacity;
    expansion_count_++;

    return true;
}
