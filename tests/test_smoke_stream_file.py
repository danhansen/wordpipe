from __future__ import annotations

import argparse
from pathlib import Path
import unittest

from scripts.smoke_stream_file import build_command, summarize_events


class SmokeStreamFileTests(unittest.TestCase):
    def test_flatpak_command_exports_wav_directory(self) -> None:
        args = argparse.Namespace(
            flatpak=True,
            command="scripts/wordpipe-dev",
            num_threads=2,
            flush_chunks=3,
        )

        command = build_command(
            args,
            Path("/home/user/.var/app/dev.wordpipe.Wordpipe/data/wordpipe/models/model"),
            Path("/tmp/smoke/input.wav"),
        )

        self.assertEqual(command[:2], ["flatpak", "run"])
        self.assertIn("--filesystem=/tmp/smoke:ro", command)
        self.assertIn(
            "--filesystem=/home/user/.var/app/dev.wordpipe.Wordpipe/data/wordpipe/models/model:ro",
            command,
        )
        self.assertIn("dev.wordpipe.Wordpipe", command)
        self.assertIn("stream-file-test", command)

    def test_summarize_events_reports_final_commit_and_metrics(self) -> None:
        summary = summarize_events(
            "\n".join(
                [
                    '{"event":"partial","text":"hello","data":{}}',
                    '{"event":"stats","text":"hello","data":{"real_time_factor":0.4,"audio_seconds":1.2,"decode_seconds":0.5}}',
                    '{"event":"commit","text":"hello world","data":{}}',
                ]
            )
        )

        self.assertEqual(summary["partials"], 1)
        self.assertEqual(summary["commits"], 1)
        self.assertEqual(summary["commit_text"], "hello world")
        self.assertEqual(summary["real_time_factor"], 0.4)


if __name__ == "__main__":
    unittest.main()
