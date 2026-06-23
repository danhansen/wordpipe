from __future__ import annotations

import unittest

from wordpipe.app import UiEvent, UiTranscriptSink


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


if __name__ == "__main__":
    unittest.main()
