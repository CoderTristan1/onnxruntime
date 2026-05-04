#pragma once

#include <string>
#include <vector>

struct ProviderCapabilities {
    bool supports_training = false;
    bool supports_fp16 = false;
    bool supports_bf16 = false;
    bool supports_quantization = false;

    std::vector<std::string> supported_ops;
};
