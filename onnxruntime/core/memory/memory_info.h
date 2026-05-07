#pragma once

#include <string>
#include <cstdint>

namespace onnxruntime {
namespace memory {

enum class MemoryDomain : uint8_t {
    kUnkown = 0,
    kHost,
    kDevice,
    kPinned,
};

enum class DeviceType : uint8_t {
    kUnknown = 0,
    kCPU,
    kCUDA,
    KRocm,
    KDirectML,
};

enum class AllocationClass : uint8_t {
    kUnkown = 0,
    kDefault,
    kWorkspace,
    kActivation,
    kWeight,
};

struct MemoryInfo {
    MemoryDomain domain{MemoryDomain::kUnkown};
    DeviceType device_type{DeviceType::kUnknown};
    int device_id{-1};
    AllocationClass allocation_class{AllocationClass::kUnkown};
    std::string tag;
};

}
}
