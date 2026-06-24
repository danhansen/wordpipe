from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wordpipe.config import load_config, save_model_profile
from wordpipe.models import DEFAULT_NEMO_SOURCE_REPO, default_model_root


class ConfigTests(unittest.TestCase):
    def test_missing_config_returns_defaults(self) -> None:
        config = load_config(Path("/tmp/wordpipe-definitely-missing.toml"))

        self.assertIsNone(config.model_dir)
        self.assertEqual(config.asr_runtime, "parakeet")
        self.assertIsNone(config.asr_worker_path)
        self.assertEqual(config.model_profile, "fast")
        self.assertEqual(config.model_root, default_model_root())
        self.assertEqual(config.nemo_source, DEFAULT_NEMO_SOURCE_REPO)
        self.assertEqual(config.overlay, "gtk")
        self.assertEqual(config.mode, "toggle")
        self.assertFalse(config.insert_partial_text)
        self.assertEqual(config.num_threads, 2)
        self.assertEqual(config.queue_seconds, 10.0)

    def test_loads_config_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "\n".join(
                    [
                        'model_dir = "/models/nemotron"',
                        'model_profile = "compact"',
                        'model_root = "/models/wordpipe"',
                        'nemo_source = "/models/source.nemo"',
                        'asr_runtime = "sherpa"',
                        'asr_worker_path = "/tmp/worker"',
                        'provider = "cpu"',
                        "num_threads = 4",
                        'overlay = "gtk"',
                        'mode = "toggle"',
                        'shortcut = "CTRL+ALT+D"',
                        "spoken_punctuation = false",
                        "dry_run_insertion = true",
                        "insert_partial_text = true",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.model_dir, Path("/models/nemotron"))
        self.assertEqual(config.model_profile, "compact")
        self.assertEqual(config.model_root, Path("/models/wordpipe"))
        self.assertEqual(config.nemo_source, "/models/source.nemo")
        self.assertEqual(config.asr_runtime, "sherpa")
        self.assertEqual(config.asr_worker_path, Path("/tmp/worker"))
        self.assertEqual(config.num_threads, 4)
        self.assertEqual(config.overlay, "gtk")
        self.assertEqual(config.mode, "toggle")
        self.assertEqual(config.shortcut, "CTRL+ALT+D")
        self.assertFalse(config.spoken_punctuation)
        self.assertTrue(config.dry_run_insertion)
        self.assertTrue(config.insert_partial_text)

    def test_save_model_profile_updates_existing_config_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                'model_profile = "fast"\nnum_threads = 4\n',
                encoding="utf-8",
            )

            saved = save_model_profile("compact", path)
            config = load_config(path)

        self.assertEqual(saved, path)
        self.assertEqual(config.model_profile, "compact")
        self.assertEqual(config.num_threads, 4)

    def test_save_model_profile_appends_missing_config_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("num_threads = 4", encoding="utf-8")

            save_model_profile("compact", path)
            text = path.read_text(encoding="utf-8")
            config = load_config(path)

        self.assertIn('model_profile = "compact"', text)
        self.assertEqual(config.model_profile, "compact")
        self.assertEqual(config.num_threads, 4)


if __name__ == "__main__":
    unittest.main()
