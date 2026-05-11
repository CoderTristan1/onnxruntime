#include "onnxruntime/core/graph/contrib_ops/contrib_defs.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {

static const char* FusedBiasGelu_ver1_doc =
    "Fused Bias + GELU (+ optional scaling) operator for training. "
    "Performs Y = GELU(X + Bias) * scale. "
    "Supports FP16, BF16, and FP32. "
    "Approximate = 1 uses tanh approximation; 0 uses erf variant.";

void RegisterFusedBiasGeluSchema() {
  ONNX_CONTRIB_OPERATOR_SCHEMA(FusedBiasGelu)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(FusedBiasGelu_ver1_doc)
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "Bias", "Bias tensor (broadcastable to X)", "T")
      .Output(0, "Y", "Output tensor", "T")
      .Attr("approximate",
            "Approximation mode: 0 = erf, 1 = tanh",
            AttributeProto::INT,
            static_cast<int64_t>(1))
      .Attr("scale",
            "Optional multiplicative scale applied after GELU",
            AttributeProto::FLOAT,
            1.0f)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(bfloat16)", "tensor(float)"},
          "Constrain input and output types to float16, bfloat16, float");
}

}
}
