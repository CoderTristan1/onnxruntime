#include "core/optimizer/gradient_builder_registry.h"

#include "core/optimizer/gradient_builder_base.h"
#include "core/optimizer/gradient_fused_bias_gelu.h"

namespace onnxruntime {

GradientBuilderRegistry::GradientBuilderRegistry() = default;

GradientBuilderRegistry::~GradientBuilderRegistry() = default;

void GradientBuilderRegistry::Register(
    const std::string& op_type,
    GradientBuilderFactory factory) {
  auto it = registry_.find(op_type);
  if (it != registry_.end()) {
    // overwrite existing registration
    it->second = std::move(factory);
  } else {
    registry_.emplace(op_type, std::move(factory));
  }
}

std::unique_ptr<GradientBuilderBase> GradientBuilderRegistry::Create(
    const std::string& op_type,
    const GradientBuilderBase::OpInfo& info) const {
  auto it = registry_.find(op_type);
  if (it == registry_.end()) {
    return nullptr;
  }
  return it->second(info);
}

void RegisterGradientBuilders(GradientBuilderRegistry& gradient_registry) {

  gradient_registry.Register(
      "FusedBiasGelu",
      [](const GradientBuilderBase::OpInfo& info) {
        return std::make_unique<FusedBiasGeluGradBuilder>(info);
      });
}

}
