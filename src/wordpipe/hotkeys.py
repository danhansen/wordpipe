from __future__ import annotations

from dataclasses import dataclass
import itertools
import re
from typing import Callable, Literal, Protocol

from .portal_dbus import GioPortalProxy, variant_options
from .probe import GLOBAL_SHORTCUTS_IFACE


HotkeyMode = Literal["hold", "toggle"]
ShortcutHandler = Callable[[], None]


class HotkeyLoop(Protocol):
    def run(self, on_activate: ShortcutHandler, on_deactivate: ShortcutHandler) -> None:
        ...


@dataclass(frozen=True)
class HotkeyConfig:
    mode: HotkeyMode = "hold"
    shortcut_id: str = "dictate"
    description: str = "Start Wordpipe dictation"
    preferred_trigger: str = "CTRL+ALT+space"


class HotkeyStateMachine:
    def __init__(
        self,
        mode: HotkeyMode,
        start: ShortcutHandler,
        stop: ShortcutHandler,
        is_listening: Callable[[], bool],
    ) -> None:
        self._mode = mode
        self._start = start
        self._stop = stop
        self._is_listening = is_listening

    def activate(self) -> None:
        if self._mode == "hold":
            self._start()
            return
        if self._is_listening():
            self._stop()
        else:
            self._start()

    def deactivate(self) -> None:
        if self._mode == "hold":
            self._stop()


class ManualHotkeyLoop:
    def __init__(self, input_lines, output) -> None:  # type: ignore[no-untyped-def]
        self._input_lines = input_lines
        self._output = output

    def run(self, on_activate: ShortcutHandler, on_deactivate: ShortcutHandler) -> None:
        print("wordpipe manual hotkey commands: down, up, toggle, quit", file=self._output)
        for raw in self._input_lines:
            command = raw.strip().lower()
            if command in {"quit", "exit"}:
                return
            if command in {"down", "toggle"}:
                on_activate()
            elif command == "up":
                on_deactivate()
            elif command:
                print(f"wordpipe: ignoring unknown manual command {command!r}", file=self._output)


class GlobalShortcutsPortalLoop:
    _tokens = itertools.count(1)

    def __init__(self, config: HotkeyConfig) -> None:
        self._config = config
        self._shortcuts = GioPortalProxy(GLOBAL_SHORTCUTS_IFACE)
        self._session_handle: str | None = None

    def run(self, on_activate: ShortcutHandler, on_deactivate: ShortcutHandler) -> None:
        from gi.repository import GLib

        self._open_session()
        loop = GLib.MainLoop()

        def activated(parameters: tuple[object, ...], _object_path: str) -> None:
            session_handle, shortcut_id, _timestamp, _options = parameters
            if str(session_handle) == self._session_handle and str(shortcut_id) == self._config.shortcut_id:
                on_activate()

        def deactivated(parameters: tuple[object, ...], _object_path: str) -> None:
            session_handle, shortcut_id, _timestamp, _options = parameters
            if str(session_handle) == self._session_handle and str(shortcut_id) == self._config.shortcut_id:
                on_deactivate()

        subscriptions = [
            self._shortcuts.subscribe_signal(GLOBAL_SHORTCUTS_IFACE, "Activated", activated),
            self._shortcuts.subscribe_signal(GLOBAL_SHORTCUTS_IFACE, "Deactivated", deactivated),
        ]
        try:
            loop.run()
        finally:
            for subscription in subscriptions:
                self._shortcuts.unsubscribe_signal(subscription)
            self.close()

    def close(self) -> None:
        if self._session_handle is None:
            return
        try:
            self._shortcuts.close_session(self._session_handle)
        finally:
            self._session_handle = None

    def _open_session(self) -> None:
        create = self._request(
            "CreateSession",
            "(a{sv})",
            {
                "handle_token": self._token("create"),
                "session_handle_token": self._token("session"),
            },
        )
        self._session_handle = str(create["session_handle"])
        self._request(
            "BindShortcuts",
            "(oa(sa{sv})sa{sv})",
            self._session_handle,
            self._shortcut_array(),
            "",
            {"handle_token": self._token("bind")},
        )

    def _shortcut_array(self) -> list[tuple[str, dict[str, object]]]:
        return [
            (
                self._config.shortcut_id,
                variant_options(
                    {
                        "description": self._config.description,
                        "preferred_trigger": self._config.preferred_trigger,
                    }
                ),
            )
        ]

    def _request(self, method: str, signature: str, *args: object) -> dict[str, object]:
        return self._shortcuts.request(
            method,
            signature,
            self._variant_args(args),
            app_id_error=(
                "GNOME GlobalShortcuts requires Wordpipe to be launched as a desktop "
                "application. Install the desktop entry with "
                "`scripts/install-wordpipe-desktop` and start it with "
                "`gtk-launch dev.wordpipe.Wordpipe`."
            ),
        )

    def _variant_args(self, args: tuple[object, ...]) -> tuple[object, ...]:
        converted: list[object] = []
        for arg in args:
            if isinstance(arg, dict):
                converted.append(variant_options(arg))
            else:
                converted.append(arg)
        return tuple(converted)

    def _token(self, prefix: str) -> str:
        raw = f"wordpipe_{prefix}_{next(self._tokens)}"
        return re.sub(r"[^A-Za-z0-9_]", "_", raw)
