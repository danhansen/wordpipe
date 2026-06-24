from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest import mock

from wordpipe.app import (
    AppModelSetup,
    UiEvent,
    UiTranscriptSink,
    WordpipeApp,
    _summarize_progress,
    profile_status_text,
)
from wordpipe.config import load_config
from wordpipe.daemon import DaemonConfig
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


class FakeGLib:
    @staticmethod
    def idle_add(callback, *args):  # type: ignore[no-untyped-def]
        callback(*args)
        return 1


class FakeButton:
    def __init__(self) -> None:
        self.sensitive_values: list[bool] = []

    def set_sensitive(self, sensitive: bool) -> None:
        self.sensitive_values.append(sensitive)


class FakeDropdown:
    def __init__(self, selected: int) -> None:
        self._selected = selected

    def get_selected(self) -> int:
        return self._selected


class AppControllerStateTests(unittest.TestCase):
    def test_open_controller_reenables_dictate_button_after_setup_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            config = DaemonConfig(model_dir=model_dir, dry_run_insertion=True)
            app = WordpipeApp(config)
            button = FakeButton()
            app._glib = FakeGLib()
            app._toggle_button = button

            with mock.patch("wordpipe.app.DictationController") as controller_cls:
                controller_cls.return_value.open.return_value = None

                self.assertFalse(app._open_controller())

        self.assertEqual(button.sensitive_values, [True])
        controller_cls.return_value.open.assert_called_once_with()

    def test_profile_change_persists_selected_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            config_path.write_text('model_profile = "fast"\n', encoding="utf-8")
            app = WordpipeApp(
                None,
                model_setup=AppModelSetup(
                    model_root=root / "models",
                    model_profile="fast",
                    nemo_source="nvidia/example",
                    config_path=config_path,
                ),
            )
            app._glib = FakeGLib()

            app._profile_changed(FakeDropdown(1), None)

            config = load_config(config_path)

        self.assertEqual(config.model_profile, "compact")
        self.assertEqual(app._selected_profile, "compact")


if __name__ == "__main__":
    unittest.main()
