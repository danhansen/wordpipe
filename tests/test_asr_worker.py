from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from wordpipe.asr_worker import AsrWorkerConfig, _create_recognizer, discover_model_layout


class _OnlineRecognizer:
    called: tuple[str, dict[str, object]] | None = None

    @staticmethod
    def from_nemo_ctc(**kwargs: object) -> object:
        _OnlineRecognizer.called = ("nemo", kwargs)
        return object()

    @staticmethod
    def from_transducer(**kwargs: object) -> object:
        _OnlineRecognizer.called = ("transducer", kwargs)
        return object()


class ModelDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = sys.modules.get("sherpa_onnx")
        sys.modules["sherpa_onnx"] = types.SimpleNamespace(OnlineRecognizer=_OnlineRecognizer)
        _OnlineRecognizer.called = None

    def tearDown(self) -> None:
        if self._previous is None:
            sys.modules.pop("sherpa_onnx", None)
        else:
            sys.modules["sherpa_onnx"] = self._previous

    def test_single_onnx_model_uses_nemo_ctc_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "tokens.txt").write_text("a\n", encoding="utf-8")
            (model_dir / "model.int8.onnx").write_text("", encoding="utf-8")

            _create_recognizer(AsrWorkerConfig(model_dir=model_dir))
            layout = discover_model_layout(model_dir)

        self.assertIsNotNone(_OnlineRecognizer.called)
        name, kwargs = _OnlineRecognizer.called
        self.assertEqual(name, "nemo")
        self.assertEqual(layout.kind, "nemo_ctc")
        self.assertEqual(kwargs["tokens"], str(model_dir / "tokens.txt"))
        self.assertEqual(kwargs["model"], str(model_dir / "model.int8.onnx"))
        self.assertFalse(kwargs["enable_endpoint_detection"])

    def test_endpoint_detection_can_be_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "tokens.txt").write_text("a\n", encoding="utf-8")
            (model_dir / "model.int8.onnx").write_text("", encoding="utf-8")

            _create_recognizer(
                AsrWorkerConfig(model_dir=model_dir, enable_endpoint_detection=True)
            )

        self.assertIsNotNone(_OnlineRecognizer.called)
        _name, kwargs = _OnlineRecognizer.called
        self.assertTrue(kwargs["enable_endpoint_detection"])

    def test_transducer_layout_uses_transducer_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "tokens.txt").write_text("a\n", encoding="utf-8")
            (model_dir / "encoder.onnx").write_text("", encoding="utf-8")
            (model_dir / "decoder.onnx").write_text("", encoding="utf-8")
            (model_dir / "joiner.onnx").write_text("", encoding="utf-8")

            _create_recognizer(AsrWorkerConfig(model_dir=model_dir))
            layout = discover_model_layout(model_dir)

        self.assertIsNotNone(_OnlineRecognizer.called)
        name, kwargs = _OnlineRecognizer.called
        self.assertEqual(name, "transducer")
        self.assertEqual(layout.kind, "transducer")
        self.assertEqual(kwargs["encoder"], str(model_dir / "encoder.onnx"))
        self.assertEqual(kwargs["decoder"], str(model_dir / "decoder.onnx"))
        self.assertEqual(kwargs["joiner"], str(model_dir / "joiner.onnx"))


if __name__ == "__main__":
    unittest.main()
