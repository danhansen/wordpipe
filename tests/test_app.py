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
        self.active_values: list[bool] = []

    def set_sensitive(self, sensitive: bool) -> None:
        self.sensitive_values.append(sensitive)

    def set_active(self, active: bool) -> None:
        self.active_values.append(active)


class FakeLabel:
    def __init__(self) -> None:
        self.text = ""

    def set_text(self, text: str) -> None:
        self.text = text


class FakeActionRow:
    def __init__(self) -> None:
        self.subtitle = ""

    def set_subtitle(self, text: str) -> None:
        self.subtitle = text


class FakeBanner:
    def __init__(self) -> None:
        self.title = ""
        self.revealed_values: list[bool] = []

    def set_title(self, text: str) -> None:
        self.title = text

    def set_revealed(self, revealed: bool) -> None:
        self.revealed_values.append(revealed)


class FakeToastOverlay:
    def __init__(self) -> None:
        self.toasts: list[str] = []

    def add_toast(self, toast) -> None:  # type: ignore[no-untyped-def]
        self.toasts.append(toast.text)


class FakeAdw:
    class Toast:
        def __init__(self, text: str) -> None:
            self.text = text

        @classmethod
        def new(cls, text: str):  # type: ignore[no-untyped-def]
            return cls(text)


class FakeDropdown:
    def __init__(self, selected: int) -> None:
        self._selected = selected

    def get_selected(self) -> int:
        return self._selected


class AppControllerStateTests(unittest.TestCase):
    def test_set_label_updates_adwaita_action_row_subtitle(self) -> None:
        app = WordpipeApp(None)
        row = FakeActionRow()

        app._set_label(row, "Ready")

        self.assertEqual(row.subtitle, "Ready")

    def test_error_text_updates_banner_visibility(self) -> None:
        app = WordpipeApp(None)
        row = FakeActionRow()
        banner = FakeBanner()
        app._error_label = row
        app._error_banner = banner

        app._set_error_text("Portal denied")
        app._set_error_text("")

        self.assertEqual(row.subtitle, "")
        self.assertEqual(banner.title, "")
        self.assertEqual(banner.revealed_values, [True, False])

    def test_show_toast_uses_adwaita_toast_overlay(self) -> None:
        app = WordpipeApp(None)
        overlay = FakeToastOverlay()
        app._adw = FakeAdw
        app._toast_overlay = overlay

        app._show_toast("Compact model installed")

        self.assertEqual(overlay.toasts, ["Compact model installed"])

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

    def test_open_controller_closes_failed_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            config = DaemonConfig(model_dir=model_dir, dry_run_insertion=True)
            app = WordpipeApp(config)
            button = FakeButton()
            app._glib = FakeGLib()
            app._toggle_button = button

            with mock.patch("wordpipe.app.DictationController") as controller_cls:
                controller_cls.return_value.open.side_effect = RuntimeError("portal failed")

                self.assertFalse(app._open_controller())

        controller_cls.return_value.close.assert_called_once_with()
        self.assertIsNone(app._controller)
        self.assertEqual(button.sensitive_values[-1], False)

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

    def test_profile_change_rebuilds_config_when_no_controller_is_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "models"
            fast_runtime_dir = profile_runtime_dir(model_root, "fast")
            compact_runtime_dir = profile_runtime_dir(model_root, "compact")
            compact_runtime_dir.mkdir(parents=True)
            (compact_runtime_dir / "tokenizer.model").write_text("", encoding="utf-8")
            (compact_runtime_dir / "encoder.ort").write_text("", encoding="utf-8")
            (compact_runtime_dir / "decoder_joint.ort").write_text("", encoding="utf-8")

            def config_for_profile(profile: str) -> DaemonConfig:
                return DaemonConfig(
                    model_dir=profile_runtime_dir(model_root, profile),
                    dry_run_insertion=True,
                )

            app = WordpipeApp(
                DaemonConfig(model_dir=fast_runtime_dir, dry_run_insertion=True),
                model_setup=AppModelSetup(
                    model_root=model_root,
                    model_profile="fast",
                    nemo_source="nvidia/example",
                ),
                controller_config_factory=config_for_profile,
            )
            app._glib = FakeGLib()
            app._status_label = FakeLabel()
            app._profile_status_label = FakeLabel()
            app._toggle_button = FakeButton()
            app._install_button = FakeButton()

            with mock.patch("wordpipe.app.DictationController") as controller_cls:
                controller_cls.return_value.open.return_value = None

                app._profile_changed(FakeDropdown(1), None)

        self.assertEqual(app._selected_profile, "compact")
        controller_config = controller_cls.call_args.args[0]
        self.assertEqual(controller_config.model_dir, compact_runtime_dir)

    def test_install_complete_does_not_report_ready_when_controller_open_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "models"
            runtime_dir = profile_runtime_dir(model_root, "compact")
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "tokenizer.model").write_text("", encoding="utf-8")
            (runtime_dir / "encoder.ort").write_text("", encoding="utf-8")
            (runtime_dir / "decoder_joint.ort").write_text("", encoding="utf-8")
            app = WordpipeApp(
                DaemonConfig(model_dir=runtime_dir, dry_run_insertion=True),
                model_setup=AppModelSetup(
                    model_root=model_root,
                    model_profile="compact",
                    nemo_source="nvidia/example",
                ),
            )
            app._glib = FakeGLib()
            app._status_label = FakeLabel()
            app._error_label = FakeLabel()
            app._profile_status_label = FakeLabel()
            app._toggle_button = FakeButton()
            app._install_button = FakeButton()

            with mock.patch("wordpipe.app.DictationController") as controller_cls:
                controller_cls.return_value.open.side_effect = RuntimeError("portal failed")

                self.assertFalse(
                    app._apply_event(UiEvent("install-complete", f"compact:{runtime_dir}"))
                )

        self.assertNotEqual(app._status_label.text, "Ready")
        self.assertIn("portal failed", app._error_label.text)

    def test_install_complete_for_previous_profile_does_not_open_selected_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "models"
            compact_runtime_dir = profile_runtime_dir(model_root, "compact")
            fast_runtime_dir = profile_runtime_dir(model_root, "fast")
            fast_runtime_dir.mkdir(parents=True)
            (fast_runtime_dir / "tokenizer.model").write_text("", encoding="utf-8")
            (fast_runtime_dir / "encoder.ort").write_text("", encoding="utf-8")
            (fast_runtime_dir / "decoder_joint.ort").write_text("", encoding="utf-8")
            app = WordpipeApp(
                DaemonConfig(model_dir=fast_runtime_dir, dry_run_insertion=True),
                model_setup=AppModelSetup(
                    model_root=model_root,
                    model_profile="fast",
                    nemo_source="nvidia/example",
                ),
            )
            app._glib = FakeGLib()
            app._status_label = FakeLabel()
            app._error_label = FakeLabel()
            app._profile_status_label = FakeLabel()
            app._toggle_button = FakeButton()
            app._install_button = FakeButton()
            app._selected_profile = "fast"

            with mock.patch("wordpipe.app.DictationController") as controller_cls:
                self.assertFalse(
                    app._apply_event(UiEvent("install-complete", f"compact:{compact_runtime_dir}"))
                )

        controller_cls.assert_not_called()
        self.assertIn("Compact model installed", app._status_label.text)

    def test_install_error_for_previous_profile_keeps_selected_profile_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_root = root / "models"
            fast_runtime_dir = profile_runtime_dir(model_root, "fast")
            fast_runtime_dir.mkdir(parents=True)
            (fast_runtime_dir / "tokenizer.model").write_text("", encoding="utf-8")
            (fast_runtime_dir / "encoder.ort").write_text("", encoding="utf-8")
            (fast_runtime_dir / "decoder_joint.ort").write_text("", encoding="utf-8")
            app = WordpipeApp(
                DaemonConfig(model_dir=fast_runtime_dir, dry_run_insertion=True),
                model_setup=AppModelSetup(
                    model_root=model_root,
                    model_profile="fast",
                    nemo_source="nvidia/example",
                ),
            )
            app._glib = FakeGLib()
            app._status_label = FakeLabel()
            app._error_label = FakeLabel()
            app._profile_status_label = FakeLabel()
            app._toggle_button = FakeButton()
            app._install_button = FakeButton()
            app._selected_profile = "fast"

            self.assertFalse(
                app._apply_event(UiEvent("install-error", "compact:RuntimeError: boom"))
            )

        self.assertIn("Compact install failed: RuntimeError: boom", app._error_label.text)
        self.assertIn("Fast: installed", app._status_label.text)
        self.assertNotEqual(app._status_label.text, "Setup required")


if __name__ == "__main__":
    unittest.main()
