from __future__ import annotations

import unittest

from wordpipe.listen_test import _format_event


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


if __name__ == "__main__":
    unittest.main()
