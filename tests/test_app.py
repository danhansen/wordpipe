from __future__ import annotations

import unittest
from pathlib import Path
import subprocess
import tempfile
from typing import Sequence
from unittest import mock

from wordpipe.app import (
    AppModelSetup,
    UiEvent,
    UiTranscriptSink,
    WordpipeApp,
    _selected_microphone_index,
    _summarize_progress,
    profile_status_text,
)
from wordpipe.config import load_config
from wordpipe.daemon import DaemonConfig
from wordpipe.models import profile_runtime_dir
from wordpipe.shortcuts import flatpak_shortcut_spec, install_shortcut


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
        self.labels: list[str] = []

    def set_sensitive(self, sensitive: bool) -> None:
        self.sensitive_values.append(sensitive)

    def set_active(self, active: bool) -> None:
        self.active_values.append(active)

    def set_label(self, label: str) -> None:
        self.labels.append(label)


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


class FakeStringList:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels

    @classmethod
    def new(cls, labels: list[str]):  # type: ignore[no-untyped-def]
        return cls(labels)


class FakeGtk:
    StringList = FakeStringList


class FakeMicrophoneDropdown:
    def __init__(self) -> None:
        self.model = None
        self.selected_values: list[int] = []
        self._selected = 0

    def set_model(self, model) -> None:  # type: ignore[no-untyped-def]
        self.model = model

    def set_selected(self, selected: int) -> None:
        self._selected = selected
        self.selected_values.append(selected)

    def get_selected(self) -> int:
        return self._selected


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


def _shortcut_runner():
    values: dict[tuple[str, str], str] = {
        (
            "org.gnome.settings-daemon.plugins.media-keys",
            "custom-keybindings",
        ): "@as []",
    }

    def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(command)
        if args[1] == "get":
            return subprocess.CompletedProcess(args, 0, values.get((args[2], args[3]), "''"), "")
        if args[1] == "set":
            values[(args[2], args[3])] = args[4]
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 2, "", "unsupported")

    return run


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

    def test_refresh_shortcut_state_updates_status_and_repair_action(self) -> None:
        app = WordpipeApp(None)
        row = FakeActionRow()
        button = FakeButton()
        app._shortcut_status_label = row
        app._shortcut_button = button
        spec = flatpak_shortcut_spec(binding="<Super>d")
        status = install_shortcut(spec, runner=_shortcut_runner())

        with mock.patch("wordpipe.app.flatpak_shortcut_spec", return_value=spec):
            with mock.patch("wordpipe.app.read_shortcut_status", return_value=status):
                self.assertFalse(app._refresh_shortcut_state())

        self.assertIn("installed: <Super>d", row.subtitle)
        self.assertEqual(button.labels, ["Repair"])

    def test_refresh_shortcut_state_reports_gsettings_failure(self) -> None:
        app = WordpipeApp(None)
        row = FakeActionRow()
        button = FakeButton()
        app._shortcut_status_label = row
        app._shortcut_button = button

        with mock.patch("wordpipe.app.read_shortcut_status", side_effect=RuntimeError("no dconf")):
            self.assertFalse(app._refresh_shortcut_state())

        self.assertIn("Shortcut status unavailable: no dconf", row.subtitle)
        self.assertEqual(button.labels, ["Retry"])

    def test_install_gnome_shortcut_surfaces_success_toast(self) -> None:
        app = WordpipeApp(None)
        row = FakeActionRow()
        banner = FakeBanner()
        button = FakeButton()
        overlay = FakeToastOverlay()
        app._adw = FakeAdw
        app._shortcut_status_label = row
        app._shortcut_button = button
        app._error_banner = banner
        app._error_label = FakeActionRow()
        app._toast_overlay = overlay
        spec = flatpak_shortcut_spec(binding="<Super>d")
        status = install_shortcut(spec, runner=_shortcut_runner())

        with mock.patch("wordpipe.app.flatpak_shortcut_spec", return_value=spec):
            with mock.patch("wordpipe.app.install_shortcut", return_value=status):
                app._install_gnome_shortcut(button)

        self.assertIn("installed: <Super>d", row.subtitle)
        self.assertEqual(button.sensitive_values, [False, True])
        self.assertEqual(overlay.toasts, ["Keyboard shortcut installed"])

    def test_refresh_microphones_populates_dropdown_and_keeps_selection(self) -> None:
        app = WordpipeApp(DaemonConfig(model_dir=Path("/models"), input_device="cpal:1"))
        row = FakeActionRow()
        dropdown = FakeMicrophoneDropdown()
        app._gtk = FakeGtk
        app._microphone_status_label = row
        app._microphone_dropdown = dropdown

        devices = [
            mock.Mock(name="Built-in", selector="cpal:0", is_default=True),
            mock.Mock(name="USB Mic", selector="cpal:1", is_default=False),
        ]
        devices[0].name = "Built-in"
        devices[1].name = "USB Mic"
        with mock.patch("wordpipe.app.list_parakeet_input_devices", return_value=devices):
            self.assertFalse(app._refresh_microphones())

        self.assertEqual(dropdown.model.labels, ["System Default", "Default: Built-in", "USB Mic"])
        self.assertEqual(dropdown.selected_values, [2])
        self.assertEqual(app._microphone_selectors, [None, "cpal:0", "cpal:1"])
        self.assertEqual(row.subtitle, "Selected input: cpal:1")

    def test_apply_input_device_persists_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            app = WordpipeApp(
                DaemonConfig(model_dir=root / "model", dry_run_insertion=True),
                model_setup=AppModelSetup(
                    model_root=root / "models",
                    model_profile="fast",
                    nemo_source="nvidia/example",
                    config_path=config_path,
                ),
            )
            row = FakeActionRow()
            app._microphone_status_label = row

            app._apply_input_device("cpal:2")
            config = load_config(config_path)

        self.assertEqual(config.input_device, "cpal:2")
        self.assertEqual(app._config.input_device, "cpal:2")
        self.assertEqual(row.subtitle, "Selected input: cpal:2")

    def test_selected_microphone_index_defaults_when_missing(self) -> None:
        self.assertEqual(_selected_microphone_index([None, "cpal:0"], "cpal:0"), 1)
        self.assertEqual(_selected_microphone_index([None, "cpal:0"], "cpal:9"), 0)

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
