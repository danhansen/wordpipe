from __future__ import annotations

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

from wordpipe.asr_worker import (
    AsrWorkerConfig,
    _create_recognizer,
    discover_model_layout,
    render_model_info,
)


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

    def test_worker_config_rejects_invalid_endpoint_rules(self) -> None:
        cases = [
            (
                {"endpoint_rule1_min_trailing_silence": 0.0},
                "endpoint_rule1_min_trailing_silence must be positive",
            ),
            (
                {"endpoint_rule2_min_trailing_silence": -1.0},
                "endpoint_rule2_min_trailing_silence must be positive",
            ),
            (
                {"endpoint_rule3_min_utterance_length": math.inf},
                "endpoint_rule3_min_utterance_length must be positive",
            ),
        ]
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    AsrWorkerConfig(model_dir=Path("/models/sherpa"), **overrides)

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

    def test_parakeet_nemotron_layout_does_not_require_sherpa_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "tokenizer.model").write_text("", encoding="utf-8")
            (model_dir / "encoder.ort").write_text("", encoding="utf-8")
            (model_dir / "decoder_joint.ort").write_text("", encoding="utf-8")
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            layout = discover_model_layout(model_dir)
            rendered = render_model_info(model_dir)

        self.assertEqual(layout.kind, "parakeet_nemotron")
        self.assertIsNone(layout.tokens)
        self.assertEqual(layout.encoder, model_dir / "encoder.ort")
        self.assertEqual(layout.decoder_joint, model_dir / "decoder_joint.ort")
        self.assertEqual(layout.tokenizer, model_dir / "tokenizer.model")
        self.assertIn('"kind": "parakeet_nemotron"', rendered)
        self.assertIn('"tokens": null', rendered)

    def test_parakeet_nemotron_layout_is_rejected_by_sherpa_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "tokenizer.model").write_text("", encoding="utf-8")
            (model_dir / "encoder.onnx").write_text("", encoding="utf-8")
            (model_dir / "decoder_joint.onnx").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Rust parakeet runtime"):
                _create_recognizer(AsrWorkerConfig(model_dir=model_dir))

    def test_asr_worker_config_rejects_invalid_runtime_values(self) -> None:
        cases = [
            ({"num_threads": 0}, "num_threads must be positive"),
            ({"sample_rate": 0}, "sample_rate must be positive"),
            ({"feature_dim": 0}, "feature_dim must be positive"),
            ({"partial_interval_seconds": 0.0}, "partial_interval_seconds must be positive"),
            ({"audio_chunk_seconds": 0.0}, "audio_chunk_seconds must be positive"),
            ({"queue_seconds": 0.0}, "queue_seconds must be positive"),
            ({"stats_interval_seconds": math.inf}, "stats_interval_seconds must be positive"),
        ]
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    AsrWorkerConfig(model_dir=Path("/models/sherpa"), **overrides)


if __name__ == "__main__":
    unittest.main()
