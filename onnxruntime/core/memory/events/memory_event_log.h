#pragma once
#include <string>
#include <vector>
#include <mutex>

namespace onnxruntime {
namespace memory {

struct MemoryEvent {
  std::string type;
  std::string detail;
};

class MemoryEventLog {
 public:
  static MemoryEventLog& Instance();

  void AddEvent(const std::string& type, const std::string& detail);
  std::vector<MemoryEvent> GetEvents() const;
  void Clear();

 private:
  MemoryEventLog() = default;

  mutable std::mutex mutex_;
  std::vector<MemoryEvent> events_;
};

}
}
