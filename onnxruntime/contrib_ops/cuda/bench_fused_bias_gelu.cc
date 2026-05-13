#include <chrono>
#include <iostream>
#include <vector>
#include <random>

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/fused_bias_gelu.h"

using namespace onnxruntime;
using namespace onnxruntime::contrib::cuda;

template <typename T>
void InitRandom(std::vector<T>& v, float scale = 1.0f) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (auto& x : v) x = static_cast<T>(dist(gen));
}

template <typename T>
void RunBenchmark(int64_t batch, int64_t hidden_size, int iters) {
  int64_t N = batch * hidden_size;

  std::vector<T> h_x(N);
  std::vector<T> h_bias(hidden_size);
  std::vector<T> h_y(N);

  InitRandom(h_x);
  InitRandom(h_bias);

  T* d_x = nullptr;
  T* d_bias = nullptr;
  T* d_y = nullptr;

  CUDA_CALL(cudaMalloc(&d_x, N * sizeof(T)));
  CUDA_CALL(cudaMalloc(&d_bias, hidden_size * sizeof(T)));
  CUDA_CALL(cudaMalloc(&d_y, N * sizeof(T)));

  CUDA_CALL(cudaMemcpy(d_x, h_x.data(), N * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_bias, h_bias.data(), hidden_size * sizeof(T), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreate(&stream));

  // Warmup
  for (int i = 0; i < 10; ++i) {
    LaunchFusedBiasGeluKernel<T>(
        stream,
        d_x,
        d_bias,
        d_y,
        N,
        hidden_size,
        /*approximate=*/1,
        /*scale=*/1.0f);
  }
  CUDA_CALL(cudaStreamSynchronize(stream));

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iters; ++i) {
    LaunchFusedBiasGeluKernel<T>(
        stream,
        d_x,
        d_bias,
        d_y,
        N,
        hidden_size,
        /*approximate=*/1,
        /*scale=*/1.0f);
  }

  CUDA_CALL(cudaStreamSynchronize(stream));
  auto end = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  double avg_ms = ms / iters;

  double bytes =
      static_cast<double>(N * sizeof(T) * 2 + hidden_size * sizeof(T));  // x + y + bias
  double gbps = (bytes / 1e9) / (avg_ms / 1e3);

  std::cout << "FusedBiasGelu<" << (sizeof(T) == 4 ? "fp32" : "fp16")
            << "> batch=" << batch
            << " hidden=" << hidden_size
            << " iters=" << iters
            << " avg=" << avg_ms << " ms"
            << " ~" << gbps << " GB/s"
            << std::endl;

  CUDA_CALL(cudaFree(d_x));
  CUDA_CALL(cudaFree(d_bias));
  CUDA_CALL(cudaFree(d_y));
  CUDA_CALL(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
  int64_t batch = 32;
  int64_t hidden = 4096;
  int iters = 200;

  if (argc >= 3) {
    batch = std::stoll(argv[1]);
    hidden = std::stoll(argv[2]);
  }
  if (argc >= 4) {
    iters = std::stoi(argv[3]);
  }

  std::cout << "Running FusedBiasGelu benchmark: batch=" << batch
            << " hidden=" << hidden
            << " iters=" << iters << std::endl;

  RunBenchmark<float>(batch, hidden, iters);

  return 0;
}
