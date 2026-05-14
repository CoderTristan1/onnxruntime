#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include <cmath>

namespace onnxruntime {
namespace test {

static float GeluTanhGrad(float x) {
  const float k0 = 0.7978845608028654f;
  const float k1 = 0.044715f;
  float x3 = x * x * x;
  float u = k0 * (x + k1 * x3);
  float t = tanhf(u);
  float du_dx = k0 * (1.0f + 3.0f * k1 * x * x);
  return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du_dx;
}

static void RunFusedBiasGeluGradTest(
    const std::vector<float>& dy,
    const std::vector<float>& x,
    const std::vector<float>& bias,
    const std::vector<float>& expected_dx,
    const std::vector<float>& expected_dbias,
    const std::vector<int64_t>& shape,
    int approximate = 1,
    float scale = 1.0f) {

  OpTester test("FusedBiasGeluGrad", 1, kMSDomain);

  test.AddInput<float>("dY", shape, dy);
  test.AddInput<float>("X", shape, x);
  test.AddInput<float>("Bias", {bias.size()}, bias);

  test.AddAttribute("approximate", approximate);
  test.AddAttribute("scale", scale);

  test.AddOutput<float>("dX", shape, expected_dx);
  test.AddOutput<float>("dBias", {bias.size()}, expected_dbias);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider});
}

TEST(FusedBiasGeluGrad, BasicBackward) {
  std::vector<float> x = {0.1f, -0.2f, 0.3f, 0.4f};
  std::vector<float> bias = {0.01f, -0.02f, 0.03f, 0.04f};
  std::vector<float> dy = {1.0f, 1.0f, 1.0f, 1.0f};

  std::vector<float> dx(4);
  std::vector<float> dbias(4, 0.0f);

  for (int i = 0; i < 4; ++i) {
    float v = x[i] + bias[i];
    float g = GeluTanhGrad(v);
    float grad = dy[i] * g;
    dx[i] = grad;
    dbias[i] += grad;
  }

  RunFusedBiasGeluGradTest(dy, x, bias, dx, dbias, {4});
}

TEST(FusedBiasGeluGrad, BroadcastBias) {
  std::vector<float> x = {1, 2, 3, 4, 5, 6};
  std::vector<float> bias = {0.5f, -0.5f};
  std::vector<float> dy = {1, 1, 1, 1, 1, 1};

  std::vector<float> dx(6);
  std::vector<float> dbias(2, 0.0f);

  for (int i = 0; i < 6; ++i) {
    float v = x[i] + bias[i % 2];
    float g = GeluTanhGrad(v);
    float grad = dy[i] * g;
    dx[i] = grad;
    dbias[i % 2] += grad;
  }

  RunFusedBiasGeluGradTest(dy, x, bias, dx, dbias, {3, 2});
}

TEST(FusedBiasGeluGrad, NonApproximateMode) {
  std::vector<float> x = {0.2f, -0.3f};
  std::vector<float> bias = {0.1f, -0.1f};
  std::vector<float> dy = {1, 1};

  std::vector<float> dx(2);
  std::vector<float> dbias(2, 0.0f);

  for (int i = 0; i < 2; ++i) {
    float v = x[i] + bias[i];
    float g = 0.5f * (1.0f + erff(v * M_SQRT1_2));
    float grad = dy[i] * g;
    dx[i] = grad;
    dbias[i] += grad;
  }

  RunFusedBiasGeluGradTest(dy, x, bias, dx, dbias, {2}, /*approximate=*/0);
}

}
}
