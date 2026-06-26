from __future__ import annotations

import contextlib
import io
import shutil
import stat
import sys
import tempfile
import tarfile
import types
import unittest
import zipfile
from unittest import mock
from pathlib import Path

from wordpipe.models import (
    DEFAULT_MODEL_REPO,
    _progress_reporter,
    build_model_profile,
    build_profile_command,
    download_prebuilt_profile,
    download_nemo_source,
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

    def test_download_prebuilt_profile_downloads_raw_hub_files(self) -> None:
        class EntryNotFoundError(Exception):
            pass

        downloaded: list[str] = []

        def hf_hub_download(**kwargs):  # type: ignore[no-untyped-def]
            filename = kwargs["filename"]
            downloaded.append(filename)
            if filename.endswith(".data"):
                raise EntryNotFoundError()
            destination = Path(kwargs["local_dir"]) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(filename, encoding="utf-8")
            return str(destination)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.dict(
                sys.modules,
                {
                    "huggingface_hub": types.SimpleNamespace(hf_hub_download=hf_hub_download),
                    "huggingface_hub.utils": types.SimpleNamespace(EntryNotFoundError=EntryNotFoundError),
                },
            ):
                source = download_prebuilt_profile(
                    profile="fast",
                    model_root=root,
                    repo_id="danhansen/example-fast",
                )

            self.assertEqual(source, root / "downloads" / "danhansen--example-fast" / "fast")
            self.assertEqual((source / "tokenizer.model").read_text(encoding="utf-8"), "tokenizer.model")
            self.assertEqual((source / "encoder.onnx").read_text(encoding="utf-8"), "encoder.onnx")
            self.assertEqual((source / "decoder_joint.onnx").read_text(encoding="utf-8"), "decoder_joint.onnx")
            self.assertIn("encoder.onnx.data", downloaded)

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

    def test_install_built_profile_preserves_existing_profile_when_force_copy_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "tokenizer.model").write_text("new-tokenizer", encoding="utf-8")
            (source / "encoder.ort").write_text("new-encoder", encoding="utf-8")
            (source / "decoder_joint.ort").write_text("new-decoder", encoding="utf-8")
            destination = profile_runtime_dir(root / "installed", "compact")
            destination.mkdir(parents=True)
            (destination / "tokenizer.model").write_text("old-tokenizer", encoding="utf-8")
            (destination / "encoder.ort").write_text("old-encoder", encoding="utf-8")
            (destination / "decoder_joint.ort").write_text("old-decoder", encoding="utf-8")

            original_copytree = shutil.copytree

            def failing_copytree(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
                original_copytree(src, dst, *args, **kwargs)
                raise RuntimeError("copy failed")

            with (
                mock.patch("wordpipe.models.shutil.copytree", side_effect=failing_copytree),
                self.assertRaisesRegex(RuntimeError, "copy failed"),
            ):
                install_built_profile(
                    source=source,
                    model_root=root / "installed",
                    profile="compact",
                    force=True,
                )

            self.assertEqual(
                (destination / "tokenizer.model").read_text(encoding="utf-8"),
                "old-tokenizer",
            )
            self.assertFalse(any(destination.parent.glob(f".{destination.name}.tmp-*")))

    def test_install_built_profile_imports_zip_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("profile/tokenizer.model", "tokenizer")
                archive.writestr("profile/encoder.ort", "encoder")
                archive.writestr("profile/decoder_joint.ort", "decoder")

            runtime_dir = install_built_profile(
                source=archive_path,
                model_root=root / "installed",
                profile="compact",
            )

            self.assertEqual(runtime_dir, profile_runtime_dir(root / "installed", "compact"))
            self.assertEqual((runtime_dir / "tokenizer.model").read_text(encoding="utf-8"), "tokenizer")

    def test_install_built_profile_removes_zip_extract_tempdir_after_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extract_dir = root / "extract"
            archive_path = root / "profile.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("profile/tokenizer.model", "tokenizer")
                archive.writestr("profile/encoder.ort", "encoder")
                archive.writestr("profile/decoder_joint.ort", "decoder")

            def make_tempdir(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                extract_dir.mkdir()
                return str(extract_dir)

            with mock.patch("tempfile.mkdtemp", side_effect=make_tempdir):
                install_built_profile(
                    source=archive_path,
                    model_root=root / "installed",
                    profile="compact",
                )

            self.assertFalse(extract_dir.exists())

    def test_install_built_profile_rejects_unsafe_zip_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.zip"
            extract_dir = root / "extract"
            outside = root / "evil.txt"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../evil.txt", "bad")
                archive.writestr("profile/tokenizer.model", "tokenizer")
                archive.writestr("profile/encoder.ort", "encoder")
                archive.writestr("profile/decoder_joint.ort", "decoder")

            def make_tempdir(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                extract_dir.mkdir()
                return str(extract_dir)

            with (
                mock.patch("tempfile.mkdtemp", side_effect=make_tempdir),
                self.assertRaisesRegex(RuntimeError, "unsafe zip member"),
            ):
                install_built_profile(
                    source=archive_path,
                    model_root=root / "installed",
                    profile="compact",
                )

            self.assertFalse(outside.exists())
            self.assertFalse(extract_dir.exists())

    def test_install_built_profile_rejects_zip_symlink_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.zip"
            extract_dir = root / "extract"
            with zipfile.ZipFile(archive_path, "w") as archive:
                link = zipfile.ZipInfo("profile/encoder.ort")
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(link, "/tmp/encoder.ort")
                archive.writestr("profile/tokenizer.model", "tokenizer")
                archive.writestr("profile/decoder_joint.ort", "decoder")

            def make_tempdir(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                extract_dir.mkdir()
                return str(extract_dir)

            with (
                mock.patch("tempfile.mkdtemp", side_effect=make_tempdir),
                self.assertRaisesRegex(RuntimeError, "unsupported zip member"),
            ):
                install_built_profile(
                    source=archive_path,
                    model_root=root / "installed",
                    profile="compact",
                )

            self.assertFalse(extract_dir.exists())

    def test_install_built_profile_imports_tar_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                _add_tar_text(archive, "profile/tokenizer.model", "tokenizer")
                _add_tar_text(archive, "profile/encoder.ort", "encoder")
                _add_tar_text(archive, "profile/decoder_joint.ort", "decoder")

            runtime_dir = install_built_profile(
                source=archive_path,
                model_root=root / "installed",
                profile="compact",
            )

            self.assertEqual(runtime_dir, profile_runtime_dir(root / "installed", "compact"))
            self.assertEqual((runtime_dir / "tokenizer.model").read_text(encoding="utf-8"), "tokenizer")

    def test_install_built_profile_rejects_unsafe_tar_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.tar"
            outside = root / "evil.txt"
            with tarfile.open(archive_path, "w") as archive:
                _add_tar_text(archive, "../evil.txt", "bad")
                _add_tar_text(archive, "profile/tokenizer.model", "tokenizer")
                _add_tar_text(archive, "profile/encoder.ort", "encoder")
                _add_tar_text(archive, "profile/decoder_joint.ort", "decoder")

            with self.assertRaisesRegex(RuntimeError, "unsafe tar member"):
                install_built_profile(
                    source=archive_path,
                    model_root=root / "installed",
                    profile="compact",
                )

            self.assertFalse(outside.exists())

    def test_install_built_profile_rejects_tar_symlink_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "profile.tar"
            with tarfile.open(archive_path, "w") as archive:
                link = tarfile.TarInfo("profile/encoder.ort")
                link.type = tarfile.SYMTYPE
                link.linkname = "/tmp/encoder.ort"
                archive.addfile(link)
                _add_tar_text(archive, "profile/tokenizer.model", "tokenizer")
                _add_tar_text(archive, "profile/decoder_joint.ort", "decoder")

            with self.assertRaisesRegex(RuntimeError, "unsupported tar member"):
                install_built_profile(
                    source=archive_path,
                    model_root=root / "installed",
                    profile="compact",
                )

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

    def test_download_nemo_source_reports_cached_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.nemo"
            source.write_text("cached", encoding="utf-8")
            events: list[str] = []

            result = download_nemo_source(str(source), progress=events.append)

        self.assertEqual(result, source)
        self.assertEqual(events, [f"Using local source model: {source}"])

    def test_build_model_profile_reports_command_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events: list[str] = []

            with mock.patch("wordpipe.models._run_with_progress") as run:
                runtime_dir = build_model_profile(
                    source=root / "source.nemo",
                    model_root=root,
                    profile="compact",
                    python=Path("/usr/bin/python3"),
                    progress=events.append,
                )

        run.assert_called_once()
        self.assertEqual(run.call_args.args[1], events.append)
        self.assertEqual(runtime_dir, profile_runtime_dir(root, "compact"))
        self.assertTrue(any("build_nemotron_wordpipe_model.py" in event for event in events))
        self.assertEqual(events[-1], f"Model profile ready: {runtime_dir}")


def _add_tar_text(archive: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    unittest.main()
