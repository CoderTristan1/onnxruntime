#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/transformer_memcpy.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/graph_transformer_registry.h"

#include "fused_bias_gelu_rewrite_rule.h"

namespace onnxruntime {

class FusedBiasGeluTransformer : public GraphTransformer {
 public:
  FusedBiasGeluTransformer()
      : GraphTransformer("FusedBiasGeluTransformer") {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override {
    ORT_UNUSED_PARAMETER(graph_level);

    auto rule = CreateFusedBiasGeluRewriteRule();
    GraphTransformerUtils::ApplyRule(*rule, graph, modified, logger);

    return Status::OK();
  }
};

std::unique_ptr<GraphTransformer> CreateFusedBiasGeluTransformer() {
  return std::make_unique<FusedBiasGeluTransformer>();
}

}
