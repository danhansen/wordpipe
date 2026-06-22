from __future__ import annotations

import sys
import threading
from typing import Protocol, TextIO


class TranscriptSink(Protocol):
    def open(self) -> None:
        ...

    def status(self, text: str) -> None:
        ...

    def partial(self, text: str) -> None:
        ...

    def commit(self, text: str) -> None:
        ...

    def error(self, text: str) -> None:
        ...

    def close(self) -> None:
        ...


class StderrTranscriptSink:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr

    def open(self) -> None:
        return

    def status(self, text: str) -> None:
        print(f"wordpipe: {text}", file=self._stream)

    def partial(self, text: str) -> None:
        print(f"partial: {text}", file=self._stream)

    def commit(self, text: str) -> None:
        print(f"commit: {text}", file=self._stream)

    def error(self, text: str) -> None:
        print(f"wordpipe error: {text}", file=self._stream)

    def close(self) -> None:
        return


class GtkTranscriptOverlay:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._loop = None
        self._app = None
        self._label = None
        self._window = None
        self._glib = None

    def open(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="wordpipe-overlay", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def status(self, text: str) -> None:
        self._set_text(text)

    def partial(self, text: str) -> None:
        self._set_text(text)

    def commit(self, text: str) -> None:
        self._set_text(text)

    def error(self, text: str) -> None:
        self._set_text(text)

    def close(self) -> None:
        if self._glib is None:
            return

        def shutdown() -> bool:
            if self._window is not None:
                self._window.destroy()
            if self._app is not None:
                self._app.quit()
            elif self._loop is not None:
                self._loop.quit()
            return False

        self._glib.idle_add(shutdown)
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        if self._run_adwaita():
            return
        self._run_gtk()

    def _run_adwaita(self) -> bool:
        try:
            import gi

            gi.require_version("Adw", "1")
            gi.require_version("Gtk", "4.0")
            from gi.repository import Adw, Gio, GLib, Gtk

            self._glib = GLib
            app = Adw.Application(
                application_id="dev.wordpipe.Wordpipe.Overlay",
                flags=Gio.ApplicationFlags.NON_UNIQUE,
            )
            self._app = app

            def activate(application) -> None:  # type: ignore[no-untyped-def]
                window = Adw.ApplicationWindow(application=application, title="Wordpipe")
                window.set_decorated(False)
                window.set_default_size(560, 96)
                window.set_resizable(False)
                window.set_opacity(0.94)
                window.add_css_class("osd")
                window.set_content(self._build_overlay_content(Gtk))
                window.present()
                self._window = window
                self._ready.set()

            app.connect("activate", activate)
            app.run([])
            return True
        except Exception as exc:  # noqa: BLE001 - optional UI must fail visibly.
            print(
                f"wordpipe adwaita overlay unavailable: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return False

    def _run_gtk(self) -> None:
        try:
            import gi

            gi.require_version("Gtk", "4.0")
            from gi.repository import GLib, Gtk

            self._glib = GLib
            self._loop = GLib.MainLoop()
            window = Gtk.Window(title="Wordpipe")
            window.set_decorated(False)
            window.set_default_size(560, 96)
            window.set_resizable(False)
            window.set_opacity(0.94)
            window.set_child(self._build_overlay_content(Gtk))
            window.present()
            self._window = window
            self._ready.set()
            self._loop.run()
        except Exception as exc:  # noqa: BLE001 - optional UI must fail visibly.
            print(f"wordpipe gtk overlay error: {type(exc).__name__}: {exc}", file=sys.stderr)
            self._ready.set()

    def _build_overlay_content(self, Gtk):  # type: ignore[no-untyped-def]
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.set_margin_top(14)
        box.set_margin_bottom(14)
        box.set_margin_start(18)
        box.set_margin_end(18)

        label = Gtk.Label(label="Wordpipe idle")
        label.set_wrap(True)
        label.set_xalign(0)
        label.add_css_class("title-3")
        box.append(label)
        self._label = label
        return box

    def _set_text(self, text: str) -> None:
        if self._glib is None or self._label is None:
            return

        def update() -> bool:
            self._label.set_text(text)
            return False

        self._glib.idle_add(update)


def make_transcript_sink(name: str) -> TranscriptSink:
    if name == "stderr":
        return StderrTranscriptSink()
    if name == "gtk":
        return GtkTranscriptOverlay()
    raise ValueError(f"unknown transcript sink: {name}")
