#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunFusedBiasGeluTest(
    const std::vector<float>& x,
    const std::vector<float>& bias,
    const std::vector<float>& expected,
    const std::vector<int64_t>& shape,
    int approximate = 1,
    float scale = 1.0f) {

  OpTester test("FusedBiasGelu", 1, kMSDomain);

  test.AddInput<float>("X", shape, x);
  test.AddInput<float>("Bias", {bias.size()}, bias);

  test.AddAttribute("approximate", approximate);
  test.AddAttribute("scale", scale);

  test.AddOutput<float>("Y", shape, expected);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider});
}

TEST(FusedBiasGelu, BasicForward) {
  std::vector<float> x = {0.1f, -0.2f, 0.3f, 0.4f};
  std::vector<float> bias = {0.01f, -0.02f, 0.03f, 0.04f};

  std::vector<float> expected = {
      0.5f * (0.1f + 0.01f) * (1.0f + tanhf(0.79788456f * ((0.1f + 0.01f) + 0.044715f * powf((0.1f + 0.01f), 3)))),
      0.5f * (-0.2f - 0.02f) * (1.0f + tanhf(0.79788456f * ((-0.2f - 0.02f) + 0.044715f * powf((-0.2f - 0.02f), 3)))),
      0.5f * (0.3f + 0.03f) * (1.0f + tanhf(0.79788456f * ((0.3f + 0.03f) + 0.044715f * powf((0.3f + 0.03f), 3)))),
      0.5f * (0.4f + 0.04f) * (1.0f + tanhf(0.79788456f * ((0.4f + 0.04f) + 0.044715f * powf((0.4f + 0.04f), 3))))
  };

  RunFusedBiasGeluTest(x, bias, expected, {4});
}

TEST(FusedBiasGelu, BroadcastBias) {
  std::vector<float> x = {1, 2, 3, 4, 5, 6};
  std::vector<float> bias = {0.5f, -0.5f};

  std::vector<float> expected(6);
  for (int i = 0; i < 6; ++i) {
    float v = x[i] + bias[i % 2];
    expected[i] = 0.5f * v * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)));
  }

  RunFusedBiasGeluTest(x, bias, expected, {3, 2});
}

TEST(FusedBiasGelu, NonApproximateMode) {
  std::vector<float> x = {0.2f, -0.3f};
  std::vector<float> bias = {0.1f, -0.1f};

  std::vector<float> expected(2);
  for (int i = 0; i < 2; ++i) {
    float v = x[i] + bias[i];
    expected[i] = 0.5f * v * (1.0f + erff(v * M_SQRT1_2));
  }

  RunFusedBiasGeluTest(x, bias, expected, {2}, /*approximate=*/0);
}

}
}
