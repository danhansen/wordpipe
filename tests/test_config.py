from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wordpipe.config import load_config


class ConfigTests(unittest.TestCase):
    def test_missing_config_returns_defaults(self) -> None:
        config = load_config(Path("/tmp/wordpipe-definitely-missing.toml"))

        self.assertIsNone(config.model_dir)
        self.assertEqual(config.mode, "hold")
        self.assertEqual(config.num_threads, 2)
        self.assertEqual(config.queue_seconds, 10.0)

    def test_loads_config_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "\n".join(
                    [
                        'model_dir = "/models/nemotron"',
                        'provider = "cpu"',
                        "num_threads = 4",
                        'overlay = "gtk"',
                        'mode = "toggle"',
                        'shortcut = "CTRL+ALT+D"',
                        "spoken_punctuation = false",
                        "dry_run_insertion = true",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.model_dir, Path("/models/nemotron"))
        self.assertEqual(config.num_threads, 4)
        self.assertEqual(config.overlay, "gtk")
        self.assertEqual(config.mode, "toggle")
        self.assertEqual(config.shortcut, "CTRL+ALT+D")
        self.assertFalse(config.spoken_punctuation)
        self.assertTrue(config.dry_run_insertion)


if __name__ == "__main__":
    unittest.main()
