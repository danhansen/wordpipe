from __future__ import annotations

import unittest

from wordpipe.listen_test import _format_device, _format_event


class ListenTestTests(unittest.TestCase):
    def test_formats_partial_event_with_rtf(self) -> None:
        line = _format_event(
            {
                "event": "partial",
                "text": "hello",
                "data": {
                    "real_time_factor": 0.2,
                    "audio_seconds": 1.0,
                    "decode_seconds": 0.2,
                    "elapsed_seconds": 1.1,
                    "dropped_audio_chunks": 0,
                },
            }
        )

        self.assertIn("partial", line)
        self.assertIn("rtf=0.2", line)
        self.assertIn("hello", line)

    def test_formats_stats_event_with_rms(self) -> None:
        line = _format_event(
            {
                "event": "stats",
                "text": "current words",
                "data": {
                    "real_time_factor": 0.2,
                    "audio_seconds": 1.0,
                    "decode_seconds": 0.2,
                    "elapsed_seconds": 1.1,
                    "last_rms": 0.01,
                    "peak_rms": 0.02,
                    "dropped_audio_chunks": 0,
                },
            }
        )

        self.assertIn("stats", line)
        self.assertIn("partial", line)
        self.assertIn("rtf=0.2", line)
        self.assertIn("rms=0.01", line)
        self.assertIn("current words", line)

    def test_formats_listening_device(self) -> None:
        self.assertIn(
            "Built-in",
            _format_device({"input_device": {"name": "Built-in", "requested": 13}}),
        )


if __name__ == "__main__":
    unittest.main()
