#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"

namespace onnxruntime {

class FusedBiasGeluRewriteRule : public RewriteRule {
 public:
  FusedBiasGeluRewriteRule() : RewriteRule("FusedBiasGeluRewriteRule") {}

  std::vector<std::string> TargetOpTypes() const override {
    return {"Mul"};  // final op in the unfused chain
  }

  bool SatisfyCondition(const Graph& graph, const Node& node) const override {
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Mul", {7, 13}, kOnnxDomain))
      return false;

    const Node* gelu = graph_utils::GetInputNode(node, 0);
    if (!gelu || !graph_utils::IsSupportedOptypeVersionAndDomain(*gelu, "Gelu", {1}, kOnnxDomain))
      return false;

    const Node* add = graph_utils::GetInputNode(*gelu, 0);
    if (!add || !graph_utils::IsSupportedOptypeVersionAndDomain(*add, "Add", {7, 13}, kOnnxDomain))
      return false;


    return true;
  }

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) const override {
    Node* mul = &node;
    Node* gelu = graph.GetNode(mul->InputNodesBegin()->Index());
    Node* add = graph.GetNode(gelu->InputNodesBegin()->Index());

    const NodeArg* X = add->InputDefs()[0];
    const NodeArg* Bias = add->InputDefs()[1];
    const NodeArg* ScaleInput = mul->InputDefs()[1];

    float scale_value = 1.0f;
    if (graph_utils::NodeArgIsConstant(graph, *ScaleInput)) {
      const TensorProto* tp = graph_utils::GetConstantInitializer(graph, ScaleInput->Name());
      if (tp && tp->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        scale_value = tp->float_data_size() > 0 ? tp->float_data(0) : 1.0f;
      }
    }

    Node& fused = graph.AddNode(
        graph.GenerateNodeName("FusedBiasGelu"),
        "FusedBiasGelu",
        "Fused Bias + GELU + Scale",
        {X, Bias},
        {mul->MutableOutputDefs()[0]},
        nullptr,
        kMSDomain);

    fused.AddAttribute("approximate", static_cast<int64_t>(1));  // default tanh
    fused.AddAttribute("scale", scale_value);

    graph_utils::RemoveNodeOutputEdges(graph, *mul);
    graph.RemoveNode(mul->Index());
    graph.RemoveNode(gelu->Index());
    graph.RemoveNode(add->Index());

    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
    return Status::OK();
  }
};

std::unique_ptr<RewriteRule> CreateFusedBiasGeluRewriteRule() {
  return std::make_unique<FusedBiasGeluRewriteRule>();
}

}
