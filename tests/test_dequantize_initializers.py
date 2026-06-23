from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _load_script(name: str):
    script = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_output_initializer_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [],
        "initializer-output-test",
        [],
        [helper.make_tensor_value_info("encoded_len", TensorProto.INT64, [1])],
        initializer=[
            numpy_helper.from_array(np.asarray([7], dtype=np.int64), name="encoded_len"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="unused_len"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])


class DequantizeInitializerPruningTests(unittest.TestCase):
    def test_matmul_pruning_preserves_graph_output_initializers(self) -> None:
        module = _load_script("dequantize_nemotron_matmul_blocks.py")
        model = _make_output_initializer_model()

        self.assertEqual(module.prune_unused_initializers(model), 1)

        initializers = {initializer.name for initializer in model.graph.initializer}
        self.assertEqual(initializers, {"encoded_len"})
        onnx.checker.check_model(model)

    def test_conv_pruning_preserves_graph_output_initializers(self) -> None:
        module = _load_script("dequantize_nemotron_conv_blocks.py")
        model = _make_output_initializer_model()

        self.assertEqual(module.prune_unused_initializers(model), 1)

        initializers = {initializer.name for initializer in model.graph.initializer}
        self.assertEqual(initializers, {"encoded_len"})
        onnx.checker.check_model(model)


if __name__ == "__main__":
    unittest.main()
