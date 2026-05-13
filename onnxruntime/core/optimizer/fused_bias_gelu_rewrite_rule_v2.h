#pragma once

#include "core/optimizer/rewrite_rule.h"
#include "core/graph/graph.h"
#include "core/graph/node_arg.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {

class FusedBiasGeluRewriteRuleV2 : public RewriteRule {
 public:
  FusedBiasGeluRewriteRuleV2()
      : RewriteRule("FusedBiasGeluRewriteRuleV2") {}

  const std::vector<std::string>& TargetOpTypes() const override {
    static const std::vector<std::string> target_ops = {"Gelu", "GeluGrad"};
    return target_ops;
  }

  bool SatisfyCondition(const Graph& graph, const Node& gelu_node) const override {
    // Must have exactly 1 input
    if (gelu_node.InputDefs().size() != 1) return false;

    const NodeArg* x_arg = gelu_node.InputDefs()[0];
    if (!x_arg) return false;

    // Check upstream Add
    const Node* add_node = graph.GetProducerNode(x_arg->Name());
    if (!add_node || add_node->OpType() != "Add") return false;

    // Check Add inputs: X + Bias
    if (add_node->InputDefs().size() != 2) return false;

    const NodeArg* a = add_node->InputDefs()[0];
    const NodeArg* b = add_node->InputDefs()[1];
    if (!a || !b) return false;

    // Bias must be initializer or broadcastable
    const TensorShape* bias_shape = b->Shape();
    if (!bias_shape) return false;

    // Bias must be 1D or last-dim broadcast
    if (bias_shape->NumDimensions() > 1) return false;

    // Check type support
    auto* type_proto = a->TypeAsProto();
    if (!type_proto) return false;

    const auto& t = type_proto->tensor_type().elem_type();
    if (!(t == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
          t == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
          t == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16)) {
      return false;
    }

    return true;
  }

  Status Apply(Graph& graph, Node& gelu_node, bool& modified, const logging::Logger&) const override {
    const NodeArg* x_arg = gelu_node.InputDefs()[0];
    const Node* add_node = graph.GetProducerNode(x_arg->Name());
    const NodeArg* a = add_node->InputDefs()[0];
    const NodeArg* b = add_node->InputDefs()[1];

    Node& fused = graph.AddNode(
        graph.GenerateNodeName("FusedBiasGelu"),
        "FusedBiasGelu",
        "Fused Bias + Gelu fusion",
        {a, b},
        {gelu_node.MutableOutputDefs()[0]},
        nullptr,
        kMSDomain);

    // Copy attributes from Gelu
    for (const auto& attr : gelu_node.GetAttributes()) {
      fused.AddAttribute(attr.first, attr.second);
    }

    // Remove old nodes
    graph_utils::RemoveNodeOutputEdges(graph, gelu_node);
    graph.RemoveNode(gelu_node.Index());
    graph.RemoveNode(add_node->Index());

    modified = true;
    return Status::OK();
  }
};

std::unique_ptr<RewriteRule> CreateFusedBiasGeluRewriteRuleV2() {
  return std::make_unique<FusedBiasGeluRewriteRuleV2>();
}

}
