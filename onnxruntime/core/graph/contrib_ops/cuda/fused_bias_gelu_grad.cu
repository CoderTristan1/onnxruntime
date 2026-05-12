#include "contrib_ops/cuda/fused_bias_gelu_grad.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ inline float GeluTanhGrad(float x) {
  const float k0 = 0.7978845608028654f;
  const float k1 = 0.044715f;
  float x3 = x * x * x;
  float u = k0 * (x + k1 * x3);
  float t = tanhf(u);
  float du_dx = k0 * (1.0f + 3.0f * k1 * x * x);
  return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du_dx;
}

template <typename T>
__global__ void FusedBiasGeluGradKernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    const T* __restrict__ bias,
    T* __restrict__ dx,
    T* __restrict__ dbias,
    int64_t N,
    int64_t bias_stride,
    int approximate,
    float scale) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N) return;

  int64_t b_idx = idx % bias_stride;

  float xv = static_cast<float>(x[idx]) + static_cast<float>(bias[b_idx]);
  float dyv = static_cast<float>(dy[idx]);

  float g = GeluTanhGrad<T>(xv);  // we default to tanh grad
  if (!approximate) {
    // simple erf-based grad approximation: d/dx gelu(x) ≈ 0.5 * (1 + erf(x / sqrt(2)))
    const float kAlpha = M_SQRT1_2;
    g = 0.5f * (1.0f + erff(kAlpha * xv));
  }

  float local = dyv * g * scale;

  dx[idx] = static_cast<T>(local);

  atomicAdd(reinterpret_cast<float*>(&dbias[b_idx]),
            local);  // dbias is reduced over N
}

template <typename T>
void LaunchFusedBiasGeluGradKernel(
    cudaStream_t stream,
    const void* dy,
    const void* x,
    const void* bias,
    void* dx,
    void* dbias,
    int64_t N,
    int64_t bias_stride,
    int approximate,
    float scale) {
  int threads = 256;
  int blocks = static_cast<int>((N + threads - 1) / threads);

  FusedBiasGeluGradKernel<T><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const T*>(dy),
      reinterpret_cast<const T*>(x),
      reinterpret_cast<const T*>(bias),
      reinterpret_cast<T*>(dx),
      reinterpret_cast<T*>(dbias),
      N,
      bias_stride,
      approximate,
      scale);
}

template void LaunchFusedBiasGeluGradKernel<float>(
    cudaStream_t, const void*, const void*, const void*, void*, void*, int64_t, int64_t, int, float);
template void LaunchFusedBiasGeluGradKernel<half>(
    cudaStream_t, const void*, const void*, const void*, void*, void*, int64_t, int64_t, int, float);
template void LaunchFusedBiasGeluGradKernel<BFloat16>(
    cudaStream_t, const void*, const void*, const void*, void*, void*, int64_t, int64_t, int, float);

}
}
}
