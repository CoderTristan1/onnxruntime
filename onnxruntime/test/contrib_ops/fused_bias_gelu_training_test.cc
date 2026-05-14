#include "gtest/gtest.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/asserts.h"
#include "test/providers/provider_test_utils.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/graph_transformer_level.h"

namespace onnxruntime {
namespace test {

static void BuildTrainingGraph(Model& model) {
  auto& graph = model.MainGraph();

  // Inputs
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto* X = graph.GetOrCreateNodeArg("X", &tensor_float);
  auto* Bias = graph.GetOrCreateNodeArg("Bias", &tensor_float);
  auto* dY = graph.GetOrCreateNodeArg("dY", &tensor_float);

  // Forward fused op
  auto* Y = graph.GetOrCreateNodeArg("Y", &tensor_float);
  Node& fused = graph.AddNode(
      "FusedBiasGeluNode",
      "FusedBiasGelu",
      "Forward fused op",
      {X, Bias},
      {Y},
      nullptr,
      kMSDomain);

  fused.AddAttribute("approximate", static_cast<int64_t>(1));

  // Backward op
  auto* dX = graph.GetOrCreateNodeArg("dX", &tensor_float);
  auto* dBias = graph.GetOrCreateNodeArg("dBias", &tensor_float);

  Node& grad = graph.AddNode(
      "FusedBiasGeluGradNode",
      "FusedBiasGeluGrad",
      "Backward fused op",
      {dY, X, Bias},
      {dX, dBias},
      nullptr,
      kMSDomain);

  grad.AddAttribute("approximate", static_cast<int64_t>(1));

  ASSERT_STATUS_OK(graph.Resolve());
}

TEST(FusedBiasGeluTraining, EndToEndTrainingGraph) {
  Model model("FusedBiasGeluTrainingTest", false, DefaultLoggingManager().DefaultLogger());
  BuildTrainingGraph(model);

  GraphTransformerManager manager{TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()};

  auto rule = onnxruntime::CreateFusedBiasGeluTransformerV2();
  ASSERT_STATUS_OK(manager.Register(std::move(rule), TransformerLevel::Level1));

  ASSERT_STATUS_OK(manager.ApplyAll(model.MainGraph(), DefaultLoggingManager().DefaultLogger()));

  SessionOptions so;
  so.session_logid = "FusedBiasGeluTrainingTest";
  so.graph_optimization_level = TransformerLevel::Level1;

  InferenceSession session{so, DefaultLoggingManager().DefaultLogger()};
  ASSERT_STATUS_OK(session.Load(model.ToProto()));
  ASSERT_STATUS_OK(session.Initialize());

  std::vector<float> x = {0.1f, -0.2f, 0.3f, 0.4f};
  std::vector<float> bias = {0.01f, -0.02f, 0.03f, 0.04f};
  std::vector<float> dy = {1, 1, 1, 1};

  OrtValue x_val, bias_val, dy_val;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(), {4}, x, &x_val);
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(), {4}, bias, &bias_val);
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(), {4}, dy, &dy_val);

  NameMLValMap feeds;
  feeds["X"] = x_val;
  feeds["Bias"] = bias_val;
  feeds["dY"] = dy_val;

  std::vector<std::string> outputs = {"Y", "dX", "dBias"};
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, outputs, &fetches));

  ASSERT_EQ(fetches.size(), 3u);
  ASSERT_TRUE(fetches[0].IsTensor());
  ASSERT_TRUE(fetches[1].IsTensor());
  ASSERT_TRUE(fetches[2].IsTensor());
}

}
}
