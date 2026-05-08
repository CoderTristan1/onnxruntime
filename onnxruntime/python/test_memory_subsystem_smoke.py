import onnxruntime as ort
import numpy as np

def test_runtime_initializes():

    sess = ort.InferenceSession(
        ort.get_available_providers()[0]
    )
    assert sess is not None


def test_inference_still_works():

    import onnx
    from onnx import helper, TensorProto

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])

    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "TestGraph", [X], [Y])
    model = helper.make_model(graph)


    model_bytes = model.SerializeToString()

    sess = ort.InferenceSession(model_bytes)
    output = sess.run(None, {"X": np.array([1.0], dtype=np.float32)})

    assert output[0][0] == 1.0


def test_providers_still_register():

    providers = ort.get_available_providers()
    assert isinstance(providers, list)
    assert len(providers) > 0
