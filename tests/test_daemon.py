from __future__ import annotations

import unittest
from pathlib import Path

from wordpipe.daemon import AsrProcess, DaemonConfig, DictationController, format_committed_text


class FakeKeyboard:
    def __init__(self) -> None:
        self.inserted: list[str] = []

    def open(self) -> None:
        return

    def insert_text(self, text: str) -> None:
        self.inserted.append(text)

    def close(self) -> None:
        return


class FakeTranscript:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def open(self) -> None:
        return

    def status(self, text: str) -> None:
        self.events.append(("status", text))

    def partial(self, text: str) -> None:
        self.events.append(("partial", text))

    def commit(self, text: str) -> None:
        self.events.append(("commit", text))

    def error(self, text: str) -> None:
        self.events.append(("error", text))

    def close(self) -> None:
        return


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

    def test_streaming_partials_insert_only_appended_suffixes(self) -> None:
        keyboard = FakeKeyboard()
        transcript = FakeTranscript()
        controller = DictationController(
            DaemonConfig(model_dir=Path("/models/parakeet"), insert_partial_text=True),
            keyboard,
            transcript,
        )

        controller._handle_event({"event": "partial", "text": "hello"})
        controller._handle_event({"event": "partial", "text": "hello world"})
        controller._handle_event({"event": "commit", "text": "hello world"})

        self.assertEqual(keyboard.inserted, ["hello", " world", " "])
        self.assertEqual(transcript.events[-1], ("commit", "hello world "))

    def test_streaming_partials_do_not_duplicate_rewritten_text(self) -> None:
        keyboard = FakeKeyboard()
        transcript = FakeTranscript()
        controller = DictationController(
            DaemonConfig(model_dir=Path("/models/parakeet"), insert_partial_text=True),
            keyboard,
            transcript,
        )

        controller._handle_event({"event": "partial", "text": "hello world"})
        controller._handle_event({"event": "partial", "text": "yellow world today"})

        self.assertEqual(keyboard.inserted, ["hello world"])
        self.assertIn(
            ("status", "partial changed before already-inserted text; waiting for append"),
            transcript.events,
        )


if __name__ == "__main__":
    unittest.main()
