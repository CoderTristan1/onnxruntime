#include "core_telemetry.h"
#include <iostream>

void CoreTelemetry::Log(const std::string& msg) {
    std::cout << "[ORT] " << msg << std::endl;
}
