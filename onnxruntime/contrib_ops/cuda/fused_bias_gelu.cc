#include "fused_bias_gelu.h"
#include "cuda_common.h"
#include "cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

FusedBiasGelu::FusedBiasGelu(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t approximate = 1;
  if (info.GetAttr<int64_t>("approximate", &approximate).IsOK()) {
    approximate_ = static_cast<int>(approximate);
  } else {
    approximate_ = 1;
  }

  float scale = 1.0f;
  if (info.GetAttr<float>("scale", &scale).IsOK()) {
    scale_ = scale;
  } else {
    scale_ = 1.0f;
  }
}

Status FusedBiasGelu::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  ORT_ENFORCE(X != nullptr && B != nullptr, "Inputs must not be null");

  const TensorShape& x_shape = X->Shape();
  const TensorShape& b_shape = B->Shape();

  ORT_ENFORCE(x_shape.Size() > 0, "X must be non-empty");
  ORT_ENFORCE(b_shape.Size() > 0, "Bias must be non-empty");

  const int64_t N = x_shape.Size();
  const int64_t bias_size = b_shape.Size();
  ORT_ENFORCE(bias_size <= N, "Bias size must be <= X size for broadcast");

  Tensor* Y = ctx->Output(0, x_shape);
  ORT_ENFORCE(Y != nullptr, "Output tensor Y must not be null");

  const void* x_data = X->DataRaw();
  const void* b_data = B->DataRaw();
  void* y_data = Y->MutableDataRaw();

  auto* stream = Stream();
  const auto dtype = X->GetElementType();

  switch (dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      LaunchFusedBiasGeluKernel<float>(
          stream, x_data, b_data, y_data, N, bias_size, approximate_, scale_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      LaunchFusedBiasGeluKernel<half>(
          stream, x_data, b_data, y_data, N, bias_size, approximate_, scale_);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      LaunchFusedBiasGeluKernel<BFloat16>(
          stream, x_data, b_data, y_data, N, bias_size, approximate_, scale_);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported data type for FusedBiasGelu");
  }

  return Status::OK();
}

}
}
}
