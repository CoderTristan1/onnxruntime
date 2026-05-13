#include "core/optimizer/graph_transformer_registry.h"

#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/graph_transformer_level.h"

#include "core/optimizer/constant_folding.h"
#include "core/optimizer/common_subexpression_elimination.h"
#include "core/optimizer/shape_to_initializer.h"

#include "core/optimizer/fused_bias_gelu_transformer_v2.h"

namespace onnxruntime {

void RegisterTransformersV2(GraphTransformerRegistry& registry) {
  {
    std::vector<std::unique_ptr<GraphTransformer>> level1;

    level1.emplace_back(std::make_unique<ConstantFolding>());
    level1.emplace_back(std::make_unique<CommonSubexpressionElimination>());
    level1.emplace_back(std::make_unique<ShapeToInitializer>());

    level1.emplace_back(CreateFusedBiasGeluTransformerV2());

    registry.Register(
        TransformerLevel::Level1,
        std::move(level1));
  }

  {
    std::vector<std::unique_ptr<GraphTransformer>> level2;


    registry.Register(
        TransformerLevel::Level2,
        std::move(level2));
  }
}

}
