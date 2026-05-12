#include "contrib_ops/cuda/fused_bias_gelu_grad.h"

// ...

ONNX_OPERATOR_KERNEL_EX(
    FusedBiasGeluGrad,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}),
    contrib::cuda::FusedBiasGeluGrad);
