#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

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
    float scale);

class FusedBiasGeluGrad final : public CudaKernel {
 public:
  FusedBiasGeluGrad(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

private:
 int approximate_{1};
 float scale_{1.0f};
};

}
}
}
