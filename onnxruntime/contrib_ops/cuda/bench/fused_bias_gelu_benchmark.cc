#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/providers/cuda/cuda_provider_factory.h"

using namespace onnxruntime;

static std::vector<float> RandomVector(int64_t n) {
  std::vector<float> v(n);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = dist(gen);
  return v;
}

static std::unique_ptr<Model> BuildFusedModel(int64_t hidden) {
  auto model = std::make_unique<Model>("FusedBiasGeluBench", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model->MainGraph();

  TypeProto t;
  t.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto* X = graph.GetOrCreateNodeArg("X", &t);
  auto* Bias = graph.GetOrCreateNodeArg("Bias", &t);
  auto* Y = graph.GetOrCreateNodeArg("Y", &t);

  Node& fused = graph.AddNode(
      "FusedBiasGeluNode",
      "FusedBiasGelu",
      "Benchmark fused op",
      {X, Bias},
      {Y},
      nullptr,
      kMSDomain);

  fused.AddAttribute("approximate", static_cast<int64_t>(1));

  ORT_THROW_IF_ERROR(graph.Resolve());
  return model;
}

static std::unique_ptr<Model> BuildUnfusedModel(int64_t hidden) {
  auto model = std::make_unique<Model>("UnfusedBench", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model->MainGraph();

  TypeProto t;
  t.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto* X = graph.GetOrCreateNodeArg("X", &t);
  auto* Bias = graph.GetOrCreateNodeArg("Bias", &t);
  auto* Z = graph.GetOrCreateNodeArg("Z", &t);
  auto* Y = graph.GetOrCreateNodeArg("Y", &t);

  graph.AddNode("AddNode", "Add", "X + Bias", {X, Bias}, {Z});
  Node& gelu = graph.AddNode("GeluNode", "Gelu", "Gelu(Z)", {Z}, {Y});
  gelu.AddAttribute("approximate", static_cast<int64_t>(1));

  ORT_THROW_IF_ERROR(graph.Resolve());
  return model;
}

static double BenchmarkModel(Model& model, int64_t hidden, int iters) {
  SessionOptions so;
  so.session_logid = "FusedBiasGeluBench";
  so.graph_optimization_level = TransformerLevel::Level1;

  InferenceSession session{so, DefaultLoggingManager().DefaultLogger()};
  ORT_THROW_IF_ERROR(session.Load(model.ToProto()));
  ORT_THROW_IF_ERROR(session.Initialize());

  std::vector<float> x = RandomVector(hidden);
  std::vector<float> bias = RandomVector(hidden);

  OrtValue x_val, bias_val;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(), {1, hidden}, x, &x_val);
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(), {hidden}, bias, &bias_val);

  NameMLValMap feeds;
  feeds["X"] = x_val;
  feeds["Bias"] = bias_val;

  std::vector<std::string> outputs = {"Y"};
  std::vector<OrtValue> fetches;

  for (int i = 0; i < 10; ++i) {
    session.Run(RunOptions{}, feeds, outputs, &fetches);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    session.Run(RunOptions{}, feeds, outputs, &fetches);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  return ms / iters;
}

int main() {
  std::vector<int64_t> hidden_sizes = {1024, 2048, 4096, 8192};
  int iters = 200;

  std::ofstream csv("fused_bias_gelu_bench_cpp.csv");
  csv << "hidden_size,fused_ms,unfused_ms,speedup\n";

  for (int64_t hidden : hidden_sizes) {
    auto fused = BuildFusedModel(hidden);
    auto unfused = BuildUnfusedModel(hidden);

    double fused_ms = BenchmarkModel(*fused, hidden, iters);
    double unfused_ms = BenchmarkModel(*unfused, hidden, iters);
    double speedup = unfused_ms / fused_ms;

    std::cout << "\nHidden=" << hidden << "\n";
    std::cout << "  Fused:   " << fused_ms << " ms\n";
    std::cout << "  Unfused: " << unfused_ms << " ms\n";
    std::cout << "  Speedup: " << speedup << "x\n";

    csv << hidden << "," << fused_ms << "," << unfused_ms << "," << speedup << "\n";
  }

  csv.close();
  std::cout << "\nBenchmark results saved to fused_bias_gelu_bench_cpp.csv\n";
  return 0;
}
