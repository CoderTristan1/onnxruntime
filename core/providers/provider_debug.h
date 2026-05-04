#pragma once

#include <string>

class ProviderDebug {
public:
    virtual ~ProviderDebug() = default;

    virtual std::string DumpState() const = 0;
};
