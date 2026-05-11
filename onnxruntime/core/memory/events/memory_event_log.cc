#include "memory_event_log.h"

namespace onnxruntime {
namespace memory {

MemoryEventLog& MemoryEventLog::Instance() {
  static MemoryEventLog instance;
  return instance;
}

void MemoryEventLog::AddEvent(const std::string& type, const std::string& detail) {
  std::lock_guard<std::mutex> lock(mutex_);
  events_.push_back({type, detail});
}

std::vector<MemoryEvent> MemoryEventLog::GetEvents() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return events_;
}

void MemoryEventLog::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  events_.clear();
}

}
}
