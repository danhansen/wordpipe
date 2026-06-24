from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_converter():
    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_nemotron_to_ort_format.py"
    spec = importlib.util.spec_from_file_location("convert_nemotron_to_ort_format", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OrtFormatConverterTests(unittest.TestCase):
    def test_force_does_not_remove_output_until_source_is_valid(self) -> None:
        converter = _load_converter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            (output / "keep.txt").write_text("existing", encoding="utf-8")
            (source / "encoder.onnx").write_text("encoder", encoding="utf-8")
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
            args = argparse.Namespace(
                source_dir=source,
                output_dir=output,
                optimization_level="all",
                optimization_style="fixed",
                force=True,
                save_optimized_onnx=False,
            )

            with (
                mock.patch.object(converter, "parse_args", return_value=args),
                self.assertRaisesRegex(SystemExit, "Missing required file"),
            ):
                converter.main()

            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
