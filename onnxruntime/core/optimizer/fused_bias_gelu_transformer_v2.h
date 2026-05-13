#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/rule_based_graph_transformer.h"

#include "fused_bias_gelu_rewrite_rule_v2.h"

namespace onnxruntime {

class FusedBiasGeluTransformerV2 : public RuleBasedGraphTransformer {
 public:
  FusedBiasGeluTransformerV2()
      : RuleBasedGraphTransformer("FusedBiasGeluTransformerV2", {}) {
    rules_.push_back(CreateFusedBiasGeluRewriteRuleV2());
  }

  bool ShouldOnlyApplyOnce() const override {
    return false;
  }
};

std::unique_ptr<GraphTransformer> CreateFusedBiasGeluTransformer() {
    return std::make_unique<FusedBiasGeluTransformerV2>();
}

}
