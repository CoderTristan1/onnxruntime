#include "contrib_ops/cuda/fused_bias_gelu_grad.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

FusedBiasGeluGrad::FusedBiasGeluGrad(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t approximate = 1;
  ORT_IGNORE_RETURN_VALUE(info.GetAttr<int64_t>("approximate", &approximate));
  approximate_ = static_cast<int>(approximate);

  float scale = 1.0f;
  ORT_IGNORE_RETURN_VALUE(info.GetAttr<float>("scale", &scale));
  scale_ = scale;
}

Status FusedBiasGeluGrad::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* dY = ctx->Input<Tensor>(0);
  const Tensor* X = ctx->Input<Tensor>(1);
  const Tensor* B = ctx->Input<Tensor>(2);

  ORT_ENFORCE(dY && X && B, "Inputs must not be null");

  const auto& x_shape = X->Shape();
  const auto& b_shape = B->Shape();
  const auto& dy_shape = dY->Shape();

  ORT_ENFORCE(x_shape == dy_shape, "X and dY must have same shape");
  ORT_ENFORCE(b_shape.Size() > 0, "Bias must be non-empty");

  int64_t N = x_shape.Size();
  int64_t bias_stride = b_shape.Size();

  Tensor* dX = ctx->Output(0, x_shape);
  Tensor* dBias = ctx->Output(1, b_shape);

  const void* dy_data = dY->DataRaw();
  const void* x_data = X->DataRaw();
  const void* b_data = B->DataRaw();
  void* dx_data = dX->MutableDataRaw();
  void* dbias_data = dBias->MutableDataRaw();

  auto* stream = Stream();
  auto dtype = X->GetElementType();

  switch (dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      LaunchFusedBiasGeluGradKernel<float>(
          stream, dy_data, x_data, b_data, dx_data, dbias_data, N, bias_stride, approximate_, scale_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      LaunchFusedBiasGeluGradKernel<half>(
          stream, dy_data, x_data, b_data, dx_data, dbias_data, N, bias_stride, approximate_, scale_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      LaunchFusedBiasGeluGradKernel<BFloat16>(
          stream, dy_data, x_data, b_data, dx_data, dbias_data, N, bias_stride, approximate_, scale_);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported type for FusedBiasGeluGrad");
  }

  return Status::OK();
}

}
}
}
