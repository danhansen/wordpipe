from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from wordpipe.app import UiEvent, UiTranscriptSink, _summarize_progress, profile_status_text
from wordpipe.models import profile_runtime_dir


class UiTranscriptSinkTests(unittest.TestCase):
    def test_transcript_sink_maps_events_for_ui(self) -> None:
        events: list[UiEvent] = []
        sink = UiTranscriptSink(events.append)

        sink.open()
        sink.status("metrics: rtf=0.42 audio=1.00s")
        sink.partial("hello")
        sink.commit("hello world")
        sink.error("boom")
        sink.close()

        self.assertEqual(
            [(event.kind, event.text) for event in events],
            [
                ("status", "Starting Wordpipe"),
                ("metrics", "rtf=0.42 audio=1.00s"),
                ("partial", "hello"),
                ("commit", "hello world"),
                ("error", "boom"),
                ("status", "Closed"),
            ],
        )


class AppModelSetupTests(unittest.TestCase):
    def test_profile_status_reports_install_state_and_runtime_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = profile_runtime_dir(root, "compact")

            missing = profile_status_text(root, "compact")
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "tokenizer.model").write_text("", encoding="utf-8")
            (runtime_dir / "encoder.ort").write_text("", encoding="utf-8")
            (runtime_dir / "decoder_joint.ort").write_text("", encoding="utf-8")
            installed = profile_status_text(root, "compact")

        self.assertIn("Compact: not installed", missing)
        self.assertIn(str(runtime_dir), missing)
        self.assertIn("Compact: installed", installed)

    def test_progress_summary_keeps_recent_tail_for_long_build_output(self) -> None:
        message = "x" * 200

        summary = _summarize_progress(message)

        self.assertEqual(len(summary), 120)
        self.assertTrue(summary.startswith("..."))
        self.assertTrue(summary.endswith("x" * 20))


if __name__ == "__main__":
    unittest.main()
