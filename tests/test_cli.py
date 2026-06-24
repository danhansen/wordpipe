from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wordpipe.cli import _cmd_model_install, _resolve_model_dir, build_parser
from wordpipe.config import WordpipeConfig
from wordpipe.models import DEFAULT_NEMO_SOURCE_FILENAME, profile_runtime_dir


def _args(
    *,
    model_dir: str | None = None,
    model_profile: str | None = None,
    model_root: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        model_dir=model_dir,
        model_profile=model_profile,
        model_root=model_root,
    )


def _install_marker(model_root: Path, profile: str) -> Path:
    runtime_dir = profile_runtime_dir(model_root, profile)
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
    (runtime_dir / "encoder.onnx").write_text("encoder", encoding="utf-8")
    return runtime_dir


class CliModelResolutionTests(unittest.TestCase):
    def test_voice_keyboard_parser_accepts_profile_and_shortcut(self) -> None:
        args = build_parser().parse_args(
            [
                "voice-keyboard",
                "--model-profile",
                "compact",
                "--shortcut",
                "CTRL+ALT+D",
                "--overlay",
                "gtk",
            ]
        )

        self.assertEqual(args.command, "voice-keyboard")
        self.assertEqual(args.model_profile, "compact")
        self.assertEqual(args.shortcut, "CTRL+ALT+D")
        self.assertEqual(args.overlay, "gtk")

    def test_explicit_model_dir_wins(self) -> None:
        config = WordpipeConfig(model_dir=None, model_profile="fast")

        self.assertEqual(
            _resolve_model_dir(_args(model_dir="/models/manual"), config),
            Path("/models/manual"),
        )

    def test_config_profile_resolves_installed_runtime_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = _install_marker(root, "fast")
            config = WordpipeConfig(model_dir=None, model_profile="fast", model_root=root)

            self.assertEqual(_resolve_model_dir(_args(), config), expected)

    def test_cli_profile_override_resolves_other_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = _install_marker(root, "compact")
            config = WordpipeConfig(model_dir=None, model_profile="fast", model_root=root)

            self.assertEqual(
                _resolve_model_dir(_args(model_profile="compact"), config),
                expected,
            )

    def test_missing_selected_profile_points_to_install_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = WordpipeConfig(model_dir=None, model_profile="fast", model_root=Path(tmp))

            with self.assertRaises(SystemExit) as raised:
                _resolve_model_dir(_args(model_profile="compact"), config)

        self.assertIn("wordpipe model-install --profile compact", str(raised.exception))

    def test_model_install_download_cache_follows_model_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected_source = root / "sources" / DEFAULT_NEMO_SOURCE_FILENAME
            args = argparse.Namespace(
                config=None,
                profile="compact",
                model_root=str(root),
                source="nvidia/example",
                source_output=None,
                python="/venv/bin/python",
                force=False,
                force_source=False,
                dry_run=False,
            )

            with (
                mock.patch("wordpipe.models.download_nemo_source", return_value=expected_source) as download,
                mock.patch(
                    "wordpipe.models.build_model_profile",
                    return_value=Path("/models/runtime"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(_cmd_model_install(args), 0)

        download.assert_called_once_with(
            "nvidia/example",
            expected_source,
            force=False,
        )


if __name__ == "__main__":
    unittest.main()
