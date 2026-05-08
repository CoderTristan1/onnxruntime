#pragma once
#include <string>
#include <unordered_map>

namespace onnxruntime {
namespace memory {

class MemoryRegistry {
 public:
  void RegisterDomain(const std::string& domain);
  bool Isregistered(const std::string& domain) const;

  private:
   std::unordered_map<std::string, bool> domains_;
};

}
}
