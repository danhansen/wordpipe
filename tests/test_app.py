from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from wordpipe.app import UiEvent, UiTranscriptSink, profile_status_text
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


if __name__ == "__main__":
    unittest.main()
