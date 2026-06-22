from __future__ import annotations

import io
import unittest

from wordpipe.transcript import StderrTranscriptSink, make_transcript_sink


class TranscriptTests(unittest.TestCase):
    def test_stderr_sink_formats_messages(self) -> None:
        stream = io.StringIO()
        sink = StderrTranscriptSink(stream)

        sink.status("ready")
        sink.partial("hello")
        sink.commit("hello ")
        sink.error("broken")

        self.assertEqual(
            stream.getvalue().splitlines(),
            [
                "wordpipe: ready",
                "partial: hello",
                "commit: hello ",
                "wordpipe error: broken",
            ],
        )

    def test_factory_rejects_unknown_sink(self) -> None:
        with self.assertRaises(ValueError):
            make_transcript_sink("nope")


if __name__ == "__main__":
    unittest.main()
