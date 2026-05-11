#pragma once

#include "op_kernel.h"
#include "cuda_kernel.h"
#include "cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class FusedBiasGelu final : public CudaKernel {
 public:
  FusedBiasGelu(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

private:
 int approximate_;
 float scale_;
};

template <typename T>
void LaunchFusedBiasGeluKernel(
    cudaStream_t stream,
    const void* x,
    const void* bias,
    void* y,
    int64_t N,
    int64_t bias_stride,
    int approximate,
    float scale);
}
}
}
