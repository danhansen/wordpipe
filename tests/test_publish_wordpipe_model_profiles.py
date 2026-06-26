from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_wordpipe_model_profiles.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("publish_wordpipe_model_profiles", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PublishWordpipeModelProfilesTests(unittest.TestCase):
    def test_copy_profile_files_uses_hub_root_layout(self) -> None:
        module = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "local-fast-output"
            output = root / "release"
            source.mkdir()
            (source / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
            (source / "encoder.onnx").write_text("encoder", encoding="utf-8")
            (source / "encoder.onnx.data").write_text("encoder-data", encoding="utf-8")
            (source / "decoder_joint.onnx").write_text("decoder", encoding="utf-8")

            module.validate_publish_source(source)
            copied = module.copy_profile_files(
                source,
                output,
                force=False,
            )

            self.assertEqual({path.name for path in copied}, {"tokenizer.model", "encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx"})
            self.assertEqual((output / "tokenizer.model").read_text(encoding="utf-8"), "tokenizer")
            self.assertEqual((output / "encoder.onnx").read_text(encoding="utf-8"), "encoder")
            self.assertEqual((output / "encoder.onnx.data").read_text(encoding="utf-8"), "encoder-data")
            self.assertEqual((output / "decoder_joint.onnx").read_text(encoding="utf-8"), "decoder")

    def test_validate_publish_source_rejects_ort_runtime_cache(self) -> None:
        module = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "compact-ort-cache"
            source.mkdir()
            (source / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
            (source / "encoder.ort").write_text("encoder", encoding="utf-8")
            (source / "decoder_joint.ort").write_text("decoder", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "Publish the ONNX profile directory"):
                module.validate_publish_source(source)

    def test_model_card_includes_hub_metadata_and_attribution(self) -> None:
        module = _load_script()

        card = module.render_model_card(
            "danhansen/wordpipe-nemotron-fast-fp32-projected",
            ("fast",),
        )

        self.assertIn("language:\n- multilingual", card)
        self.assertIn("license: openmdw-1.1", card)
        self.assertIn("library_name: onnx", card)
        self.assertIn("pipeline_tag: automatic-speech-recognition", card)
        self.assertIn("base_model: nvidia/nemotron-3.5-asr-streaming-0.6b", card)
        self.assertIn("NVIDIA is the upstream model developer", card)
        self.assertIn("This repository publishes the `fast` Wordpipe profile", card)
        self.assertIn("wordpipe model-install --profile fast", card)
        self.assertIn("Do not read the upstream", card)
        self.assertIn("MODEL_SPEC.md", card)
        self.assertIn("runtime ABI assumptions", card)

    def test_model_card_rejects_multiple_profiles(self) -> None:
        module = _load_script()

        with self.assertRaisesRegex(ValueError, "one profile per repo"):
            module.render_model_card("danhansen/example", ("fast", "compact"))

    def test_model_spec_documents_runtime_constraints(self) -> None:
        module = _load_script()

        spec = module.render_model_spec(("fast", "compact"))

        self.assertIn("processed_signal=[1, 128, 65]", spec)
        self.assertIn("cache_len=56", spec)
        self.assertIn("cache_key_layer_N", spec)
        self.assertIn("projected_current_key_layer_N", spec)
        self.assertIn("caller, not the graph, rolls the projected K/V cache", spec)
        self.assertIn("scripts/build_nemotron_wordpipe_model.py", spec)

    def test_copy_reproducibility_scripts(self) -> None:
        module = _load_script()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            module.copy_reproducibility_scripts(output_dir, force=False)

            for name in module.REPRODUCIBILITY_SCRIPTS:
                self.assertTrue((output_dir / "scripts" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
