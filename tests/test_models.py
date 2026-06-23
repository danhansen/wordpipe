from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from wordpipe.models import (
    DEFAULT_MODEL_REPO,
    _progress_reporter,
    build_profile_command,
    make_download_plan,
    model_file_url,
    profile_runtime_dir,
)


class ModelDownloadTests(unittest.TestCase):
    def test_model_file_url(self) -> None:
        self.assertEqual(
            model_file_url(DEFAULT_MODEL_REPO, "tokens.txt"),
            f"https://huggingface.co/{DEFAULT_MODEL_REPO}/resolve/main/tokens.txt",
        )

    def test_download_plan_default_directory(self) -> None:
        plan = make_download_plan(Path("models"))

        self.assertEqual(plan.repo_id, DEFAULT_MODEL_REPO)
        self.assertEqual(plan.model_dir, Path("models") / DEFAULT_MODEL_REPO.split("/")[-1])
        self.assertIn("encoder.int8.onnx", plan.files)
        self.assertNotIn("test_wavs/en.wav", plan.files)

    def test_download_plan_can_include_test_wavs(self) -> None:
        plan = make_download_plan(Path("models"), include_test_wavs=True)

        self.assertIn("test_wavs/en.wav", plan.files)

    def test_progress_reporter_is_callable(self) -> None:
        reporter = _progress_reporter(Path("model.onnx"))

        with contextlib.redirect_stderr(io.StringIO()):
            reporter(0, 8192, 100)

    def test_fast_profile_build_command(self) -> None:
        command = build_profile_command(
            source=Path("/models/source.nemo"),
            model_root=Path("/models/wordpipe"),
            profile="fast",
            python=Path("/venv/bin/python"),
            force=True,
        )

        self.assertEqual(command[0], "/venv/bin/python")
        self.assertIn("--profile", command)
        self.assertIn("fp32-projected", command)
        self.assertNotIn("--emit-ort-format", command)
        self.assertIn("--force", command)
        self.assertEqual(
            profile_runtime_dir(Path("/models/wordpipe"), "fast"),
            Path("/models/wordpipe/nemotron-wordpipe-fast-fp32-projected"),
        )

    def test_compact_profile_build_command_emits_ort_format(self) -> None:
        command = build_profile_command(
            source=Path("/models/source.nemo"),
            model_root=Path("/models/wordpipe"),
            profile="compact",
            python=Path("/venv/bin/python"),
        )

        self.assertIn("--profile", command)
        self.assertIn("compact-fixed-shape", command)
        self.assertIn("--emit-ort-format", command)
        self.assertEqual(
            profile_runtime_dir(Path("/models/wordpipe"), "compact"),
            Path("/models/wordpipe/nemotron-wordpipe-compact-fixed-shape-ort-format"),
        )


if __name__ == "__main__":
    unittest.main()
