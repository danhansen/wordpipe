from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_script(name: str):
    script = Path(__file__).resolve().parents[1] / "scripts" / name
    module_name = name.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_partial_model_dir(path: Path) -> None:
    path.mkdir()
    (path / "encoder.onnx").write_text("encoder", encoding="utf-8")
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.model").write_text("tokenizer", encoding="utf-8")


class ExperimentWrapperTests(unittest.TestCase):
    def test_slim_force_does_not_remove_output_until_source_is_valid(self) -> None:
        slim = _load_script("slim_nemotron_onnx.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            _write_partial_model_dir(source)
            output.mkdir()
            (output / "keep.txt").write_text("existing", encoding="utf-8")
            args = argparse.Namespace(
                source_dir=source,
                output_dir=output,
                python=Path(sys.executable),
                force=True,
                shape_infer=False,
                size_threshold=1_048_576,
                skip_constant_folding=False,
                skip_graph_fusion=False,
                encoder_only=False,
            )

            with (
                mock.patch.object(slim, "parse_args", return_value=args),
                self.assertRaisesRegex(SystemExit, "Missing required file"),
            ):
                slim.main()

            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "existing")

    def test_olive_force_does_not_remove_output_until_source_is_valid(self) -> None:
        olive = _load_script("run_olive_onnx_pass.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            _write_partial_model_dir(source)
            output.mkdir()
            (output / "keep.txt").write_text("existing", encoding="utf-8")
            args = argparse.Namespace(
                source_dir=source,
                output_dir=output,
                pass_name="peephole",
                pass_config=None,
                component=None,
                force=True,
            )

            with (
                mock.patch.object(olive, "parse_args", return_value=args),
                self.assertRaisesRegex(SystemExit, "Missing required file"),
            ):
                olive.main()

            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
