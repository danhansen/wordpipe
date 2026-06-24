from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .daemon import DaemonConfig, DictationController
from .insertion import DryRunKeyboardBackend, PortalKeyboardBackend


@dataclass(frozen=True)
class UiEvent:
    kind: str
    text: str


UiEventCallback = Callable[[UiEvent], None]


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
    def __init__(self, config: DaemonConfig | None, setup_error: str | None = None) -> None:
        self._config = config
        self._setup_error = setup_error
        self._controller: DictationController | None = None
        self._glib = None
        self._gtk = None
        self._app = None
        self._window = None
        self._status_label = None
        self._partial_label = None
        self._commit_label = None
        self._metrics_label = None
        self._error_label = None
        self._toggle_button = None
        self._button_image = None
        self._button_label = None
        self._updating_button = False

    def run(self) -> int:
        import gi

        try:
            gi.require_version("Adw", "1")
            gi.require_version("Gtk", "4.0")
            from gi.repository import Adw, Gio, GLib, Gtk

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
        toolbar.add_top_bar(header)
        toolbar.set_content(self._build_content(Gtk))
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

    def _build_content(self, Gtk):  # type: ignore[no-untyped-def]
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

        self._partial_label = self._section(root, Gtk, "Live transcript", "No speech yet")
        self._commit_label = self._section(root, Gtk, "Last committed", "Nothing committed")
        self._error_label = Gtk.Label(label="")
        self._error_label.set_xalign(0)
        self._error_label.set_wrap(True)
        self._error_label.add_css_class("error")
        root.append(self._error_label)
        return root

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
        except Exception as exc:  # noqa: BLE001 - UI must show setup failures.
            self._post_event(UiEvent("error", f"{type(exc).__name__}: {exc}"))
            self._set_button_sensitive(False)
        return False

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
            self._set_label(self._error_label, event.text)
            self._set_button_sensitive(False)
            self._set_button_state(False)
        return False

    def _set_label(self, label, text: str) -> None:  # type: ignore[no-untyped-def]
        if label is not None:
            label.set_text(text)

    def _set_button_sensitive(self, sensitive: bool) -> None:
        if self._toggle_button is not None:
            self._toggle_button.set_sensitive(sensitive)

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


def run_app(config: DaemonConfig | None, setup_error: str | None = None) -> int:
    return WordpipeApp(config, setup_error).run()
