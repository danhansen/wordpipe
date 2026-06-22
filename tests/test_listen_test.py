from __future__ import annotations

import unittest

from wordpipe.listen_test import ListenTestFormatter, _format_device, _format_event, _new_suffix


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

    def test_new_suffix_for_append_only_hypotheses(self) -> None:
        self.assertEqual(_new_suffix("The quick", "The quick brown fox"), "brown fox")
        self.assertEqual(_new_suffix("The quick", "Different"), "Different")

    def test_formatter_prints_appended_suffixes_by_default(self) -> None:
        formatter = ListenTestFormatter()

        first = formatter.format({"event": "partial", "text": "The quick"})
        second = formatter.format({"event": "partial", "text": "The quick brown fox"})

        self.assertIn("The quick", first)
        self.assertIn("brown fox", second)
        self.assertNotIn("The quick brown fox", second)


if __name__ == "__main__":
    unittest.main()
