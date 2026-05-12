#include "contrib_ops/cuda/fused_bias_gelu.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ inline T GeluErf(T x) {
  const float kAlpha = M_SQRT1_2;  // 1/sqrt(2)
  float xf = static_cast<float>(x);
  float v = 0.5f * xf * (1.0f + erff(kAlpha * xf));
  return static_cast<T>(v);
}

template <typename T>
__device__ inline T GeluTanh(T x) {
  const float k0 = 0.7978845608028654f;   // sqrt(2/pi)
  const float k1 = 0.044715f;
  float xf = static_cast<float>(x);
  float u = k0 * (xf + k1 * xf * xf * xf);
  float v = 0.5f * xf * (1.0f + tanhf(u));
  return static_cast<T>(v);
}

template <typename T>
__global__ void FusedBiasGeluKernel(
    const T* __restrict__ x,
    const T* __restrict__ bias,
    T* __restrict__ y,
    int64_t N,
    int64_t bias_stride,
    int approximate,
    float scale) {

  const int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N) return;

  const int64_t b_idx = idx % bias_stride;

  float xv = static_cast<float>(x[idx]) + static_cast<float>(bias[b_idx]);

  T gelu_out;
  if (approximate == 1) {
    gelu_out = GeluTanh<T>(static_cast<T>(xv));
  } else {
    gelu_out = GeluErf<T>(static_cast<T>(xv));
  }

  float scaled = static_cast<float>(gelu_out) * scale;
  y[idx] = static_cast<T>(scaled);
}

template <typename T>
void LaunchFusedBiasGeluKernel(
    cudaStream_t stream,
    const void* x,
    const void* bias,
    void* y,
    int64_t N,
    int64_t bias_stride,
    int approximate,
    float scale) {

  const int threads = 256;
  const int blocks = static_cast<int>((N + threads - 1) / threads);

  FusedBiasGeluKernel<T><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const T*>(x),
      reinterpret_cast<const T*>(bias),
      reinterpret_cast<T*>(y),
      N,
      bias_stride,
      approximate,
      scale);
}

template void LaunchFusedBiasGeluKernel<float>(
    cudaStream_t, const void*, const void*, void*, int64_t, int64_t, int, float);

template void LaunchFusedBiasGeluKernel<half>(
    cudaStream_t, const void*, const void*, void*, int64_t, int64_t, int, float);

template void LaunchFusedBiasGeluKernel<BFloat16>(
    cudaStream_t, const void*, const void*, void*, int64_t, int64_t, int, float);

}
}
}
