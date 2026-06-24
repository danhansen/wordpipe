from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wordpipe.daemon import (
    AsrProcess,
    DaemonConfig,
    DictationController,
    format_committed_text,
    run_signal_hotkey_daemon,
)


class FakeKeyboard:
    def __init__(self) -> None:
        self.inserted: list[str] = []

    def open(self) -> None:
        return

    def insert_text(self, text: str) -> None:
        self.inserted.append(text)

    def close(self) -> None:
        return


class OrderedKeyboard(FakeKeyboard):
    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def open(self) -> None:
        self._order.append("keyboard-open")

    def close(self) -> None:
        self._order.append("keyboard-close")


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


class OrderedTranscript(FakeTranscript):
    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self._order = order

    def open(self) -> None:
        self._order.append("transcript-open")

    def close(self) -> None:
        self._order.append("transcript-close")


class OrderedAsr:
    def __init__(self, order: list[str], *, fail_start: bool = False) -> None:
        self._order = order
        self._fail_start = fail_start

    def start(self) -> None:
        self._order.append("asr-start")
        if self._fail_start:
            raise RuntimeError("asr start failed")

    def close(self) -> None:
        self._order.append("asr-close")


class OrderedReader:
    def __init__(self, order: list[str], name: str) -> None:
        self._order = order
        self._name = name

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        self._order.append(f"{self._name}-join")


class FakeEvent:
    def __init__(self) -> None:
        self.was_set = False

    def set(self) -> None:
        self.was_set = True

    def wait(self) -> None:
        return


class FakeAsrStderr:
    def stderr_lines(self):
        yield "first warning"
        yield "second warning"


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

    def test_stderr_reader_surfaces_worker_stderr(self) -> None:
        transcript = FakeTranscript()
        controller = DictationController(
            DaemonConfig(model_dir=Path("/models/parakeet")),
            FakeKeyboard(),
            transcript,
        )
        controller._asr = FakeAsrStderr()  # type: ignore[assignment]

        controller._read_stderr()

        self.assertIn(("error", "ASR worker stderr: first warning"), transcript.events)
        self.assertIn(("error", "ASR worker stderr: second warning"), transcript.events)

    def test_close_joins_reader_threads_before_closing_transcript(self) -> None:
        order: list[str] = []
        controller = DictationController(
            DaemonConfig(model_dir=Path("/models/parakeet")),
            FakeKeyboard(),
            OrderedTranscript(order),
        )
        controller._asr = OrderedAsr(order)  # type: ignore[assignment]
        controller._reader = OrderedReader(order, "stdout")  # type: ignore[assignment]
        controller._stderr_reader = OrderedReader(order, "stderr")  # type: ignore[assignment]

        controller.close()

        self.assertEqual(
            order,
            ["asr-close", "stdout-join", "stderr-join", "transcript-close"],
        )

    def test_open_cleans_up_transcript_and_keyboard_when_asr_start_fails(self) -> None:
        order: list[str] = []
        controller = DictationController(
            DaemonConfig(model_dir=Path("/models/parakeet")),
            OrderedKeyboard(order),
            OrderedTranscript(order),
        )
        controller._asr = OrderedAsr(order, fail_start=True)  # type: ignore[assignment]

        with self.assertRaisesRegex(RuntimeError, "asr start failed"):
            controller.open()

        self.assertEqual(
            order,
            [
                "transcript-open",
                "keyboard-open",
                "asr-start",
                "asr-close",
                "keyboard-close",
                "transcript-close",
            ],
        )
        self.assertFalse(controller._opened)

    def test_signal_hotkey_pid_file_is_written_after_controller_opens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "voice-keyboard.pid"
            order: list[str] = []
            transcript = mock.Mock()

            def open_controller() -> None:
                self.assertFalse(pid_file.exists())
                order.append("open")

            def ready_status(_text: str) -> None:
                self.assertEqual(pid_file.read_text(encoding="utf-8").strip(), "1234")
                order.append("ready")

            transcript.status.side_effect = ready_status

            with (
                mock.patch("wordpipe.daemon.os.getpid", return_value=1234),
                mock.patch("wordpipe.daemon.threading.Event", return_value=FakeEvent()),
                mock.patch("wordpipe.daemon.signal.getsignal", return_value=None),
                mock.patch("wordpipe.daemon.signal.signal"),
                mock.patch("wordpipe.daemon.DictationController") as controller_cls,
            ):
                controller_cls.return_value.open.side_effect = open_controller

                self.assertEqual(
                    run_signal_hotkey_daemon(
                        DaemonConfig(model_dir=Path("/models/parakeet"), dry_run_insertion=True),
                        transcript=transcript,
                        pid_file=pid_file,
                    ),
                    0,
                )

            self.assertEqual(order, ["open", "ready"])
            self.assertIs(controller_cls.call_args.args[2], transcript)
            controller_cls.return_value.close.assert_called_once_with()
            self.assertFalse(pid_file.exists())

    def test_signal_hotkey_restores_signals_when_controller_open_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "voice-keyboard.pid"
            previous = object()

            with (
                mock.patch("wordpipe.daemon.signal.getsignal", return_value=previous),
                mock.patch("wordpipe.daemon.signal.signal") as set_signal,
                mock.patch("wordpipe.daemon.DictationController") as controller_cls,
            ):
                controller_cls.return_value.open.side_effect = RuntimeError("portal failed")

                with self.assertRaises(RuntimeError):
                    run_signal_hotkey_daemon(
                        DaemonConfig(model_dir=Path("/models/parakeet"), dry_run_insertion=True),
                        transcript=mock.Mock(),
                        pid_file=pid_file,
                    )

            self.assertEqual(set_signal.call_count, 6)
            restored = [call.args for call in set_signal.call_args_list[-3:]]
            self.assertEqual(
                restored,
                [
                    (mock.ANY, previous),
                    (mock.ANY, previous),
                    (mock.ANY, previous),
                ],
            )
            controller_cls.return_value.close.assert_called_once_with()
            self.assertFalse(pid_file.exists())

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

        self.assertEqual(keyboard.inserted, ["hello", " world"])
        self.assertEqual(transcript.events[-1], ("commit", "hello world "))

    def test_streaming_final_commit_inserts_when_no_partial_text_was_inserted(self) -> None:
        keyboard = FakeKeyboard()
        transcript = FakeTranscript()
        controller = DictationController(
            DaemonConfig(model_dir=Path("/models/parakeet"), insert_partial_text=True),
            keyboard,
            transcript,
        )

        controller._handle_event({"event": "commit", "text": "hello world"})

        self.assertEqual(keyboard.inserted, ["hello world "])
        self.assertIn(
            ("status", "no streamed text inserted; inserting final commit"),
            transcript.events,
        )

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
