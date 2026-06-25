from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from typing import Callable

from .daemon import DaemonConfig, DictationController
from .insertion import DryRunKeyboardBackend, PortalKeyboardBackend
from .models import MODEL_PROFILES, profile_installed, profile_runtime_dir


@dataclass(frozen=True)
class UiEvent:
    kind: str
    text: str


UiEventCallback = Callable[[UiEvent], None]
ControllerConfigFactory = Callable[[str], DaemonConfig]


@dataclass(frozen=True)
class AppModelSetup:
    model_root: Path
    model_profile: str
    nemo_source: str
    config_path: Path | None = None
    python: Path = Path(sys.executable)


def profile_status_text(model_root: Path, profile: str) -> str:
    runtime_dir = profile_runtime_dir(model_root, profile)
    state = "installed" if profile_installed(model_root, profile) else "not installed"
    return f"{MODEL_PROFILES[profile].title}: {state} ({runtime_dir})"


class UiTranscriptSink:
    def __init__(self, callback: UiEventCallback) -> None:
        self._callback = callback

    def open(self) -> None:
        self._callback(UiEvent("status", "Starting Wordpipe"))

    def status(self, text: str) -> None:
        if text.startswith("metrics:"):
            self._callback(UiEvent("metrics", text.removeprefix("metrics:").strip()))
        else:
            self._callback(UiEvent("status", text))

    def partial(self, text: str) -> None:
        self._callback(UiEvent("partial", text))

    def commit(self, text: str) -> None:
        self._callback(UiEvent("commit", text))

    def error(self, text: str) -> None:
        self._callback(UiEvent("error", text))

    def close(self) -> None:
        self._callback(UiEvent("status", "Closed"))


class WordpipeApp:
    def __init__(
        self,
        config: DaemonConfig | None,
        setup_error: str | None = None,
        *,
        model_setup: AppModelSetup | None = None,
        controller_config_factory: ControllerConfigFactory | None = None,
    ) -> None:
        self._config = config
        self._setup_error = setup_error
        self._model_setup = model_setup
        self._controller_config_factory = controller_config_factory
        self._controller: DictationController | None = None
        self._glib = None
        self._gtk = None
        self._adw = None
        self._app = None
        self._window = None
        self._toast_overlay = None
        self._error_banner = None
        self._status_label = None
        self._partial_label = None
        self._commit_label = None
        self._metrics_label = None
        self._error_label = None
        self._toggle_button = None
        self._button_image = None
        self._button_label = None
        self._profile_dropdown = None
        self._profile_status_label = None
        self._install_button = None
        self._install_thread: threading.Thread | None = None
        self._last_install_progress = 0.0
        self._selected_profile = model_setup.model_profile if model_setup else "fast"
        self._updating_button = False

    def run(self) -> int:
        import gi

        try:
            gi.require_version("Adw", "1")
            gi.require_version("Gtk", "4.0")
            from gi.repository import Adw, Gio, GLib, Gtk

            self._adw = Adw
            self._glib = GLib
            self._gtk = Gtk
            app = Adw.Application(
                application_id="dev.wordpipe.Wordpipe",
                flags=Gio.ApplicationFlags.FLAGS_NONE,
            )
            app.connect("activate", lambda application: self._activate_adwaita(application, Adw, Gtk))
        except (ImportError, ValueError):
            gi.require_version("Gtk", "4.0")
            from gi.repository import Gio, GLib, Gtk

            self._glib = GLib
            self._gtk = Gtk
            app = Gtk.Application(
                application_id="dev.wordpipe.Wordpipe",
                flags=Gio.ApplicationFlags.FLAGS_NONE,
            )
            app.connect("activate", lambda application: self._activate_gtk(application, Gtk))

        self._app = app
        return int(app.run([]))

    def _activate_adwaita(self, application, Adw, Gtk) -> None:  # type: ignore[no-untyped-def]
        window = Adw.ApplicationWindow(application=application, title="Wordpipe")
        window.set_default_size(760, 460)
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.set_title_widget(Gtk.Label(label="Wordpipe"))
        preferences_button = Gtk.Button.new_from_icon_name("emblem-system-symbolic")
        preferences_button.set_tooltip_text("Preferences")
        preferences_button.connect("clicked", self._show_preferences_placeholder)
        header.pack_end(preferences_button)
        toolbar.add_top_bar(header)
        self._toast_overlay = Adw.ToastOverlay()
        self._toast_overlay.set_child(self._build_content(Gtk, Adw))
        toolbar.set_content(self._toast_overlay)
        window.set_content(toolbar)
        self._present(window)

    def _activate_gtk(self, application, Gtk) -> None:  # type: ignore[no-untyped-def]
        window = Gtk.ApplicationWindow(application=application, title="Wordpipe")
        window.set_default_size(760, 460)
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        header = Gtk.HeaderBar()
        header.set_title_widget(Gtk.Label(label="Wordpipe"))
        root.append(header)
        root.append(self._build_content(Gtk))
        window.set_child(root)
        self._present(window)

    def _present(self, window) -> None:  # type: ignore[no-untyped-def]
        self._window = window
        window.connect("close-request", self._close_request)
        window.present()
        assert self._glib is not None
        self._glib.idle_add(self._open_controller)

    def _build_content(self, Gtk, Adw=None):  # type: ignore[no-untyped-def]
        if Adw is not None:
            return self._build_adwaita_content(Gtk, Adw)
        return self._build_gtk_content(Gtk)

    def _build_adwaita_content(self, Gtk, Adw):  # type: ignore[no-untyped-def]
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        clamp = Adw.Clamp()
        clamp.set_maximum_size(760)
        clamp.set_tightening_threshold(560)
        scrolled.set_child(clamp)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        root.set_margin_top(24)
        root.set_margin_bottom(24)
        root.set_margin_start(18)
        root.set_margin_end(18)
        clamp.set_child(root)

        self._error_banner = Adw.Banner(title="")
        self._error_banner.set_revealed(False)
        root.append(self._error_banner)

        dictation = Adw.PreferencesGroup(title="Dictation")
        status_row = Adw.ActionRow(title="Status", subtitle="Initializing")
        status_row.add_prefix(Gtk.Image.new_from_icon_name("audio-input-microphone-symbolic"))
        self._status_label = status_row
        self._toggle_button = Gtk.ToggleButton()
        self._toggle_button.set_tooltip_text("Start or stop dictation")
        self._button_image = Gtk.Image.new_from_icon_name("media-record-symbolic")
        self._button_label = Gtk.Label(label="Dictate")
        self._toggle_button.set_child(self._button_content(Gtk))
        self._toggle_button.connect("toggled", self._toggle_dictation)
        status_row.add_suffix(self._toggle_button)
        status_row.set_activatable_widget(self._toggle_button)
        dictation.add(status_row)

        self._metrics_label = Adw.ActionRow(title="Performance", subtitle="RTF unavailable")
        dictation.add(self._metrics_label)
        root.append(dictation)

        model = Adw.PreferencesGroup(title="Model")
        self._build_profile_row(model, Gtk, Adw)
        root.append(model)

        transcript = Adw.PreferencesGroup(title="Transcript")
        self._partial_label = Adw.ActionRow(title="Live Transcript", subtitle="No speech yet")
        transcript.add(self._partial_label)
        self._commit_label = Adw.ActionRow(title="Last Committed", subtitle="Nothing committed")
        transcript.add(self._commit_label)
        root.append(transcript)

        diagnostics = Adw.PreferencesGroup(title="Diagnostics")
        self._error_label = Adw.ActionRow(title="Last Error", subtitle="")
        diagnostics.add(self._error_label)
        root.append(diagnostics)
        return scrolled

    def _build_gtk_content(self, Gtk):  # type: ignore[no-untyped-def]
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        root.set_margin_top(18)
        root.set_margin_bottom(18)
        root.set_margin_start(18)
        root.set_margin_end(18)

        status_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        status_icon = Gtk.Image.new_from_icon_name("audio-input-microphone-symbolic")
        status_row.append(status_icon)
        status_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._status_label = Gtk.Label(label="Initializing")
        self._status_label.set_xalign(0)
        self._status_label.add_css_class("heading")
        self._metrics_label = Gtk.Label(label="RTF unavailable")
        self._metrics_label.set_xalign(0)
        self._metrics_label.add_css_class("dim-label")
        status_box.append(self._status_label)
        status_box.append(self._metrics_label)
        status_box.set_hexpand(True)
        status_row.append(status_box)
        self._toggle_button = Gtk.ToggleButton()
        self._toggle_button.set_tooltip_text("Start or stop dictation")
        self._button_image = Gtk.Image.new_from_icon_name("media-record-symbolic")
        self._button_label = Gtk.Label(label="Dictate")
        self._toggle_button.set_child(self._button_content(Gtk))
        self._toggle_button.connect("toggled", self._toggle_dictation)
        status_row.append(self._toggle_button)
        root.append(status_row)

        self._build_profile_row(root, Gtk)
        self._partial_label = self._section(root, Gtk, "Live transcript", "No speech yet")
        self._commit_label = self._section(root, Gtk, "Last committed", "Nothing committed")
        self._error_label = Gtk.Label(label="")
        self._error_label.set_xalign(0)
        self._error_label.set_wrap(True)
        self._error_label.add_css_class("error")
        root.append(self._error_label)
        return root

    def _build_profile_row(self, root, Gtk, Adw=None) -> None:  # type: ignore[no-untyped-def]
        names = tuple(MODEL_PROFILES)
        selected = names.index(self._selected_profile) if self._selected_profile in names else 0
        self._profile_dropdown = Gtk.DropDown.new_from_strings(
            [MODEL_PROFILES[name].title for name in names]
        )
        self._profile_dropdown.set_selected(selected)
        self._profile_dropdown.connect("notify::selected", self._profile_changed)
        self._install_button = Gtk.Button(label="Install")
        self._install_button.set_tooltip_text("Download and build the selected model profile")
        self._install_button.connect("clicked", self._install_selected_profile)

        if Adw is not None:
            row = Adw.ActionRow(title="Active Profile", subtitle="")
            row.add_prefix(Gtk.Image.new_from_icon_name("folder-download-symbolic"))
            row.add_suffix(self._profile_dropdown)
            row.add_suffix(self._install_button)
            self._profile_status_label = row
            root.add(row)
        else:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
            row.set_valign(Gtk.Align.CENTER)
            row.append(self._profile_dropdown)

            self._profile_status_label = Gtk.Label(label="")
            self._profile_status_label.set_xalign(0)
            self._profile_status_label.set_hexpand(True)
            self._profile_status_label.set_wrap(True)
            self._profile_status_label.add_css_class("dim-label")
            row.append(self._profile_status_label)
            row.append(self._install_button)
            root.append(row)
        self._refresh_profile_state()

    def _section(self, root, Gtk, title: str, body: str):  # type: ignore[no-untyped-def]
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        title_label = Gtk.Label(label=title)
        title_label.set_xalign(0)
        title_label.add_css_class("caption-heading")
        body_label = Gtk.Label(label=body)
        body_label.set_xalign(0)
        body_label.set_wrap(True)
        body_label.set_vexpand(True)
        body_label.set_selectable(True)
        box.append(title_label)
        box.append(body_label)
        root.append(box)
        return body_label

    def _button_content(self, Gtk):  # type: ignore[no-untyped-def]
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.append(self._button_image)
        box.append(self._button_label)
        return box

    def _open_controller(self) -> bool:
        if self._config is None and self._controller_config_factory is not None:
            try:
                self._config = self._controller_config_factory(self._selected_profile)
                self._setup_error = None
            except SystemExit as exc:
                self._setup_error = str(exc)
        if self._config is None:
            self._post_event(UiEvent("status", "Setup required"))
            if self._setup_error:
                self._post_event(UiEvent("error", self._setup_error))
            self._set_button_sensitive(False)
            return False
        keyboard = DryRunKeyboardBackend() if self._config.dry_run_insertion else PortalKeyboardBackend()
        sink = UiTranscriptSink(self._post_event)
        self._controller = DictationController(self._config, keyboard, sink)
        try:
            self._controller.open()
            self._post_event(UiEvent("status", "Ready"))
            self._set_button_sensitive(True)
        except Exception as exc:  # noqa: BLE001 - UI must show setup failures.
            self._controller.close()
            self._controller = None
            self._post_event(UiEvent("error", f"{type(exc).__name__}: {exc}"))
            self._set_button_sensitive(False)
        return False

    def _profile_changed(self, dropdown, _param) -> None:  # type: ignore[no-untyped-def]
        names = tuple(MODEL_PROFILES)
        selected = int(dropdown.get_selected())
        if selected >= len(names):
            return
        self._selected_profile = names[selected]
        self._persist_selected_profile()
        if self._controller is not None:
            self._controller.close()
            self._controller = None
            self._set_button_state(False)
            self._set_button_sensitive(False)
        if self._controller_config_factory is not None:
            self._config = None
        self._setup_error = None
        self._refresh_profile_state()
        if self._selected_profile_installed():
            self._open_controller()

    def _persist_selected_profile(self) -> None:
        if self._model_setup is None or self._model_setup.config_path is None:
            return
        try:
            from .config import save_model_profile

            save_model_profile(self._selected_profile, self._model_setup.config_path)
        except Exception as exc:  # noqa: BLE001 - profile selection should remain usable.
            self._post_event(UiEvent("error", f"Could not save model profile: {exc}"))

    def _install_selected_profile(self, _button) -> None:  # type: ignore[no-untyped-def]
        if self._model_setup is None or self._install_thread is not None:
            return
        profile = self._selected_profile
        self._set_install_sensitive(False)
        self._post_event(UiEvent("status", f"Installing {MODEL_PROFILES[profile].title} model"))
        thread = threading.Thread(
            target=self._install_profile_thread,
            args=(profile,),
            name="wordpipe-model-install",
            daemon=True,
        )
        self._install_thread = thread
        thread.start()

    def _install_profile_thread(self, profile: str) -> None:
        assert self._model_setup is not None
        try:
            from .models import build_model_profile, default_nemo_source_path, download_nemo_source

            setup = self._model_setup
            self._post_event(UiEvent("status", "Downloading source model"))
            source_path = download_nemo_source(
                setup.nemo_source,
                default_nemo_source_path(setup.model_root),
                progress=self._post_install_progress,
            )
            self._post_event(UiEvent("status", f"Building {MODEL_PROFILES[profile].title} model"))
            runtime_dir = build_model_profile(
                source=source_path,
                model_root=setup.model_root,
                profile=profile,
                python=setup.python,
                progress=self._post_install_progress,
            )
            self._post_event(UiEvent("install-complete", f"{profile}:{runtime_dir}"))
        except Exception as exc:  # noqa: BLE001 - surface setup failures in the UI.
            self._post_event(UiEvent("install-error", f"{profile}:{type(exc).__name__}: {exc}"))

    def _post_install_progress(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_install_progress < 0.5:
            return
        self._last_install_progress = now
        self._post_event(UiEvent("status", _summarize_progress(message)))

    def _toggle_dictation(self, button) -> None:  # type: ignore[no-untyped-def]
        if self._updating_button or self._controller is None:
            return
        if button.get_active():
            self._controller.start_dictation()
            self._set_button_state(True)
        else:
            self._controller.stop_dictation()
            self._set_button_state(False)

    def _post_event(self, event: UiEvent) -> None:
        if self._glib is None:
            return
        self._glib.idle_add(self._apply_event, event)

    def _apply_event(self, event: UiEvent) -> bool:
        if event.kind == "status":
            self._set_label(self._status_label, event.text)
            normalized = event.text.lower()
            if normalized in {"idle", "closed"}:
                self._set_button_state(False)
            elif "listening" in normalized:
                self._set_button_state(True)
        elif event.kind == "partial":
            self._set_label(self._partial_label, event.text or "No speech yet")
        elif event.kind == "commit":
            self._set_label(self._commit_label, event.text or "Nothing committed")
            self._set_button_state(False)
        elif event.kind == "metrics":
            self._set_label(self._metrics_label, event.text or "RTF unavailable")
        elif event.kind == "error":
            self._set_error_text(event.text)
            self._set_button_sensitive(False)
            self._set_button_state(False)
        elif event.kind == "install-complete":
            profile, _detail = _split_profile_event(event.text)
            self._install_thread = None
            self._set_error_text("")
            self._refresh_profile_state()
            if profile is not None and profile != self._selected_profile:
                self._show_toast(f"{MODEL_PROFILES[profile].title} model installed")
                self._post_event(
                    UiEvent("status", f"{MODEL_PROFILES[profile].title} model installed")
                )
                return False
            if self._selected_profile_installed():
                self._open_controller()
            else:
                self._post_event(UiEvent("status", "Setup required"))
        elif event.kind == "install-error":
            profile, detail = _split_profile_event(event.text)
            self._install_thread = None
            self._refresh_profile_state()
            if profile is not None and profile != self._selected_profile:
                self._set_error_text(f"{MODEL_PROFILES[profile].title} install failed: {detail}")
                selected_status = (
                    profile_status_text(self._model_setup.model_root, self._selected_profile)
                    if self._model_setup
                    else "Ready"
                )
                self._post_event(UiEvent("status", selected_status))
                return False
            self._set_error_text(detail)
            self._post_event(UiEvent("status", "Setup required"))
        return False

    def _set_label(self, label, text: str) -> None:  # type: ignore[no-untyped-def]
        if label is not None:
            if hasattr(label, "set_text"):
                label.set_text(text)
            elif hasattr(label, "set_subtitle"):
                label.set_subtitle(text)

    def _set_error_text(self, text: str) -> None:
        self._set_label(self._error_label, text)
        if self._error_banner is not None:
            self._error_banner.set_title(text)
            self._error_banner.set_revealed(bool(text))

    def _show_toast(self, text: str) -> None:
        if self._toast_overlay is None or self._adw is None:
            return
        self._toast_overlay.add_toast(self._adw.Toast.new(text))

    def _show_preferences_placeholder(self, _button) -> None:  # type: ignore[no-untyped-def]
        self._show_toast("Preferences are not available yet")

    def _set_button_sensitive(self, sensitive: bool) -> None:
        if self._toggle_button is not None:
            self._toggle_button.set_sensitive(sensitive)

    def _set_install_sensitive(self, sensitive: bool) -> None:
        if self._install_button is not None:
            self._install_button.set_sensitive(sensitive)

    def _refresh_profile_state(self) -> None:
        if self._model_setup is None:
            self._set_label(self._profile_status_label, "Model profiles unavailable")
            self._set_install_sensitive(False)
            return
        installed = profile_installed(self._model_setup.model_root, self._selected_profile)
        self._set_label(
            self._profile_status_label,
            profile_status_text(self._model_setup.model_root, self._selected_profile),
        )
        self._set_install_sensitive(not installed and self._install_thread is None)

    def _selected_profile_installed(self) -> bool:
        if self._model_setup is None:
            return False
        return profile_installed(self._model_setup.model_root, self._selected_profile)

    def _set_button_state(self, listening: bool) -> None:
        if self._toggle_button is None:
            return
        self._updating_button = True
        try:
            self._toggle_button.set_active(listening)
            if self._button_image is not None:
                self._button_image.set_from_icon_name(
                    "media-playback-stop-symbolic" if listening else "media-record-symbolic"
                )
            if self._button_label is not None:
                self._button_label.set_text("Stop" if listening else "Dictate")
        finally:
            self._updating_button = False

    def _close_request(self, _window) -> bool:  # type: ignore[no-untyped-def]
        if self._controller is not None:
            self._controller.close()
            self._controller = None
        if self._app is not None:
            self._app.quit()
        return False


def run_app(
    config: DaemonConfig | None,
    setup_error: str | None = None,
    *,
    model_setup: AppModelSetup | None = None,
    controller_config_factory: ControllerConfigFactory | None = None,
) -> int:
    return WordpipeApp(
        config,
        setup_error,
        model_setup=model_setup,
        controller_config_factory=controller_config_factory,
    ).run()


def _summarize_progress(message: str) -> str:
    text = message.strip()
    if not text:
        return "Installing model"
    if len(text) <= 120:
        return text
    return f"...{text[-117:]}"


def _split_profile_event(text: str) -> tuple[str | None, str]:
    profile, separator, detail = text.partition(":")
    if separator and profile in MODEL_PROFILES:
        return profile, detail
    return None, text
