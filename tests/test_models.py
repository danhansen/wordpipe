from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from wordpipe.models import (
    DEFAULT_MODEL_REPO,
    _progress_reporter,
    build_model_profile,
    build_profile_command,
    install_built_profile,
    make_download_plan,
    model_file_url,
    profile_build_dir,
    profile_runtime_dir,
    source_may_be_built_profile_archive,
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

    def test_install_built_profile_copies_runtime_dir_to_profile_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
            (source / "encoder.ort").write_text("encoder", encoding="utf-8")
            (source / "decoder_joint.ort").write_text("decoder", encoding="utf-8")
            (source / "config.json").write_text("{}", encoding="utf-8")

            runtime_dir = install_built_profile(
                source=source,
                model_root=root / "installed",
                profile="compact",
            )

            self.assertEqual(runtime_dir, profile_runtime_dir(root / "installed", "compact"))
            self.assertEqual((runtime_dir / "tokenizer.model").read_text(encoding="utf-8"), "tokenizer")
            self.assertTrue((runtime_dir / "encoder.ort").exists())
            self.assertTrue((runtime_dir / "decoder_joint.ort").exists())

    def test_nemo_source_is_not_built_profile_archive(self) -> None:
        self.assertFalse(source_may_be_built_profile_archive(Path("source.nemo")))
        self.assertTrue(source_may_be_built_profile_archive(Path("profile.tar.gz")))

    def test_build_model_profile_removes_build_dir_after_success_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = profile_build_dir(root, "compact")
            build_dir.mkdir(parents=True)
            (build_dir / "intermediate.onnx").write_text("temporary", encoding="utf-8")

            with mock.patch("subprocess.run") as run:
                runtime_dir = build_model_profile(
                    source=root / "source.nemo",
                    model_root=root,
                    profile="compact",
                    python=Path("/usr/bin/python3"),
                )

            run.assert_called_once()
            self.assertFalse(build_dir.exists())
            self.assertEqual(runtime_dir, profile_runtime_dir(root, "compact"))

    def test_build_model_profile_can_keep_build_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = profile_build_dir(root, "compact")
            build_dir.mkdir(parents=True)

            with mock.patch("subprocess.run"):
                build_model_profile(
                    source=root / "source.nemo",
                    model_root=root,
                    profile="compact",
                    python=Path("/usr/bin/python3"),
                    keep_build_dir=True,
                )

            self.assertTrue(build_dir.exists())


if __name__ == "__main__":
    unittest.main()
