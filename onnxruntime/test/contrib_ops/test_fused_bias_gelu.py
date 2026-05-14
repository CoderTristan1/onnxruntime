import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto


def build_fused_bias_gelu_model(path="fused_bias_gelu_test.onnx"):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    Bias = helper.make_tensor_value_info("Bias", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])

    fused_node = helper.make_node(
        "FusedBiasGelu",
        inputs=["X", "Bias"],
        outputs=["Y"],
        domain="com.microsoft",
        approximate=1,
        scale=1.0,
    )

    graph = helper.make_graph(
        [fused_node],
        "FusedBiasGeluGraph",
        [X, Bias],
        [Y],
    )

    model = helper.make_model(graph, producer_name="fbg_test")
    onnx.save(model, path)
    return path


def gelu_reference(x):
    # tanh-based GELU reference
    k0 = 0.7978845608028654
    k1 = 0.044715
    return 0.5 * x * (1.0 + np.tanh(k0 * (x + k1 * x**3)))


def run_test():
    model_path = build_fused_bias_gelu_model()

    sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    x = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    bias = np.array([0.01, -0.02, 0.03, 0.04], dtype=np.float32)

    expected = gelu_reference(x + bias)

    outputs = sess.run(["Y"], {"X": x, "Bias": bias})
    y = outputs[0]

    print("Input X:      ", x)
    print("Bias:         ", bias)
    print("Output Y:     ", y)
    print("Expected Y:   ", expected)
    print("Difference:   ", np.abs(y - expected))

    if np.allclose(y, expected, atol=1e-5):
        print("\n✅ FusedBiasGelu Python test PASSED")
    else:
        print("\n❌ FusedBiasGelu Python test FAILED")


if __name__ == "__main__":
    run_test()
