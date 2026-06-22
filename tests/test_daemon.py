from __future__ import annotations

import unittest
from pathlib import Path

from wordpipe.daemon import AsrProcess, DaemonConfig, format_committed_text


class DaemonTests(unittest.TestCase):
    def test_commit_formatter_strips_and_adds_space(self) -> None:
        self.assertEqual(format_committed_text(" hello "), "hello ")

    def test_commit_formatter_ignores_empty_text(self) -> None:
        self.assertEqual(format_committed_text("   "), "")

    def test_parakeet_runtime_uses_rust_worker_command(self) -> None:
        process = AsrProcess(
            DaemonConfig(
                model_dir=Path("/models/parakeet"),
                asr_runtime="parakeet",
                asr_worker_path=Path("/tmp/wordpipe-parakeet-worker"),
                num_threads=3,
            )
        )

        command = process._command()

        self.assertEqual(command[0], "/tmp/wordpipe-parakeet-worker")
        self.assertIn("--model-dir", command)
        self.assertIn("/models/parakeet", command)
        self.assertIn("--num-threads", command)
        self.assertIn("3", command)

    def test_sherpa_runtime_uses_python_worker_command(self) -> None:
        process = AsrProcess(
            DaemonConfig(model_dir=Path("/models/sherpa"), asr_runtime="sherpa")
        )

        command = process._command()

        self.assertIn("wordpipe", command)
        self.assertIn("asr-worker", command)
        self.assertIn("--provider", command)


if __name__ == "__main__":
    unittest.main()
