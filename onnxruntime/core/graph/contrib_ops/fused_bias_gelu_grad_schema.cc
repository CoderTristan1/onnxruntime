#include "contrib_defs.h"

namespace onnxruntime {
namespace contrib {

ONNX_MS_OPERATOR_SET_SCHEMA(
    FusedBiasGeluGrad,
    1,
    OpShema()
        .SetDoc("Gradient of FusedBiasGelu")
        .Input(0, "dY", "Gradient of output", "T")
        .Input(1, "X", "Input tensor", "T")
        .Input(2, "Bias", "Bias tensor", "T")
        .Output(0, "dX", "Gradient of X", "T")
        .Output(1, "dBias", "Gradient of Bias", "T")
        .Attr("approximate", "Use tanh approximation if 1, erf otherwise.", AttributeProto::INT, static_cast<int64_t>(1))
        .Attr("scale", "Scale factor used in forward.", AttributeProto::FLOAT, 1.0f)
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
            "Constrain input and output types to float/float16/bfloat16"));
}
}
