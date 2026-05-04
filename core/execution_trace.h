#pragma once

#include <string>
#include <vector>
#include <chrono>

struct ExecutionEvent {
    std::string op_type;
    std::string provider;
    long long duration_us;
    size_t memory_before;
    size_t memory_after;
};

class ExecutionTrace {
public:
    void BeginEvent(const std::string& op_type, const std::string& provider);
    void EndEvent();

    std::string ToString() const;

private:
    std::vector<ExecutionEvent> events_;
    std::chrono::high_resolution_clock::time_point start_time_;
    ExecutionEvent current_;
};
