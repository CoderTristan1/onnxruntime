#include "activation_workspace.h"

bool ActivationWorkspace::Initialize(std::size_t initial_bytes, double max_fraction) {
    return suballocator_.Initialize(initial_bytes, max_fraction);
}

void* ActivationWorkspace::Allocate(std::size_t bytes) {
    void* ptr = suballocator_.Allocate(bytes);
    if (ptr)
        allocations_this_step_++;
    return ptr;
}

void ActivationWorkspace::Free(void* ptr) {
    suballocator_.Free(ptr);
}

void ActivationWorkspace::BeginStep() {
    allocations_this_step_ = 0;
}

void ActivationWorkspace::EndStep() {
    suballocator_.ResetStep();
    allocations_this_step_ = 0;
}

ActivationWorkspaceStats ActivationWorkspace::GetStats() const {
    return {
        suballocator_.Capacity(),
        suballocator_.PeakUsage(),
        suballocator_.ExpansionCount(),
        allocations_this_step_
    };
}
