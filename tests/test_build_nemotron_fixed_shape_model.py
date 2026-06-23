from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _load_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_nemotron_fixed_shape_model.py"
    spec = importlib.util.spec_from_file_location("build_nemotron_fixed_shape_model", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FixedShapeBuilderTests(unittest.TestCase):
    def test_processed_signal_length_can_be_replaced_with_initializer(self) -> None:
        builder = _load_builder()
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Identity",
                    ["processed_signal_length"],
                    ["encoded_len"],
                    name="length_identity",
                )
            ],
            "constant-length-test",
            [
                helper.make_tensor_value_info("processed_signal_length", TensorProto.INT64, [1]),
            ],
            [
                helper.make_tensor_value_info("encoded_len", TensorProto.INT64, [1]),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])

        self.assertTrue(builder.fix_processed_signal_length(model, 65))

        input_names = {value_info.name for value_info in model.graph.input}
        initializers = {initializer.name: initializer for initializer in model.graph.initializer}
        self.assertNotIn("processed_signal_length", input_names)
        self.assertIn("processed_signal_length", initializers)
        np.testing.assert_array_equal(
            numpy_helper.to_array(initializers["processed_signal_length"]),
            np.asarray([65], dtype=np.int64),
        )
        onnx.checker.check_model(model)

    def test_processed_signal_length_replacement_is_idempotent_when_absent(self) -> None:
        builder = _load_builder()
        graph = helper.make_graph(
            [],
            "constant-length-absent-test",
            [],
            [],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])

        self.assertFalse(builder.fix_processed_signal_length(model, 65))
        self.assertFalse(model.graph.initializer)


if __name__ == "__main__":
    unittest.main()
