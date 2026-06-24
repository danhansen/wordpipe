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
        self.assertEqual(config.stream_insert_delay_seconds, 0.0)

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
                        "stream_insert_delay_seconds = 0.03",
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
        self.assertEqual(config.stream_insert_delay_seconds, 0.03)

    def test_invalid_config_value_reports_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('model_profile = "huge"\n', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, f"invalid config {path}"):
                load_config(path)

    def test_invalid_overlay_value_reports_allowed_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('overlay = "popup"\n', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "overlay must be 'stderr' or 'gtk'"):
                load_config(path)

    def test_invalid_toml_reports_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("model_profile = [\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, f"invalid config {path}"):
                load_config(path)

    def test_bool_is_not_accepted_for_integer_or_number_values(self) -> None:
        cases = [
            ("num_threads = true\n", "num_threads must be an integer"),
            ("queue_seconds = true\n", "queue_seconds must be a number"),
        ]
        for text, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "config.toml"
                    path.write_text(text, encoding="utf-8")

                    with self.assertRaisesRegex(RuntimeError, message):
                        load_config(path)

    def test_bool_is_not_accepted_for_input_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text("input_device = true\n", encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError, "input_device must be an integer index or string name"
            ):
                load_config(path)

    def test_runtime_numbers_must_be_positive(self) -> None:
        cases = [
            ("num_threads = 0\n", "num_threads must be positive"),
            ("sample_rate = 0\n", "sample_rate must be positive"),
            ("partial_interval_seconds = 0\n", "partial_interval_seconds must be positive"),
            ("audio_chunk_seconds = 0\n", "audio_chunk_seconds must be positive"),
            ("queue_seconds = 0\n", "queue_seconds must be positive"),
            ("stats_interval_seconds = 0\n", "stats_interval_seconds must be positive"),
            ("stats_interval_seconds = inf\n", "stats_interval_seconds must be positive"),
            (
                "endpoint_rule1_min_trailing_silence = 0\n",
                "endpoint_rule1_min_trailing_silence must be positive",
            ),
            (
                "endpoint_rule2_min_trailing_silence = -1\n",
                "endpoint_rule2_min_trailing_silence must be positive",
            ),
            (
                "endpoint_rule3_min_utterance_length = 0\n",
                "endpoint_rule3_min_utterance_length must be positive",
            ),
            (
                "endpoint_rule3_min_utterance_length = nan\n",
                "endpoint_rule3_min_utterance_length must be positive",
            ),
            (
                "stream_insert_delay_seconds = -0.1\n",
                "stream_insert_delay_seconds must be non-negative",
            ),
            (
                "stream_insert_delay_seconds = nan\n",
                "stream_insert_delay_seconds must be non-negative",
            ),
        ]
        for text, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "config.toml"
                    path.write_text(text, encoding="utf-8")

                    with self.assertRaisesRegex(RuntimeError, message):
                        load_config(path)

    def test_invalid_path_value_reports_config_key_name(self) -> None:
        cases = [
            ("model_dir = 1\n", "model_dir must be a string"),
            ("model_root = 1\n", "model_root must be a string"),
            ("asr_worker_path = 1\n", "asr_worker_path must be a string"),
        ]
        for text, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "config.toml"
                    path.write_text(text, encoding="utf-8")

                    with self.assertRaisesRegex(RuntimeError, message):
                        load_config(path)

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

    def test_save_model_profile_inserts_missing_key_before_toml_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "num_threads = 4\n[ui]\nmodel_profile = \"fast\"\n",
                encoding="utf-8",
            )

            save_model_profile("compact", path)
            text = path.read_text(encoding="utf-8")
            config = load_config(path)

        self.assertLess(text.index('model_profile = "compact"'), text.index("[ui]"))
        self.assertEqual(config.model_profile, "compact")
        self.assertEqual(config.num_threads, 4)


if __name__ == "__main__":
    unittest.main()
