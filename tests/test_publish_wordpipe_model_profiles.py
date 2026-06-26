from __future__ import annotations

import importlib.util
from pathlib import Path
import tarfile
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
    def test_package_profile_uses_canonical_top_level_directory(self) -> None:
        module = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "local-fast-output"
            source.mkdir()
            (source / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
            (source / "encoder.onnx").write_text("encoder", encoding="utf-8")
            (source / "decoder_joint.onnx").write_text("decoder", encoding="utf-8")
            archive = root / "fast.tar.gz"

            module.validate_publish_source(source)
            module.package_profile(
                source,
                archive,
                spec=module.profile_spec("fast"),
                force=False,
            )

            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()

            self.assertIn("nemotron-wordpipe-fast-fp32-projected/tokenizer.model", names)
            self.assertIn("nemotron-wordpipe-fast-fp32-projected/encoder.onnx", names)
            self.assertIn("nemotron-wordpipe-fast-fp32-projected/decoder_joint.onnx", names)

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
            "danhansen/wordpipe-nemotron-3.5-asr-streaming-0.6b",
            ("fast", "compact"),
        )

        self.assertIn("language:\n- multilingual", card)
        self.assertIn("license: openmdw-1.1", card)
        self.assertIn("library_name: onnx", card)
        self.assertIn("pipeline_tag: automatic-speech-recognition", card)
        self.assertIn("base_model: nvidia/nemotron-3.5-asr-streaming-0.6b", card)
        self.assertIn("NVIDIA is the upstream model developer", card)
        self.assertIn("Do not read the upstream", card)


if __name__ == "__main__":
    unittest.main()
