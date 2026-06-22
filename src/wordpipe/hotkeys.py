from __future__ import annotations

from dataclasses import dataclass
import itertools
import re
from typing import Callable, Literal, Protocol

from .probe import GLOBAL_SHORTCUTS_IFACE, PORTAL_BUS_NAME, PORTAL_OBJECT_PATH


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
        try:
            import dbus
            from dbus.mainloop.glib import DBusGMainLoop
        except ImportError as exc:
            raise RuntimeError("dbus-python is required for GlobalShortcuts") from exc

        DBusGMainLoop(set_as_default=True)
        self._dbus = dbus
        self._config = config
        self._bus = dbus.SessionBus()
        obj = self._bus.get_object(PORTAL_BUS_NAME, PORTAL_OBJECT_PATH)
        self._shortcuts = dbus.Interface(obj, GLOBAL_SHORTCUTS_IFACE)
        self._session_handle: str | None = None

    def run(self, on_activate: ShortcutHandler, on_deactivate: ShortcutHandler) -> None:
        from gi.repository import GLib

        self._open_session()
        loop = GLib.MainLoop()

        def activated(session_handle, shortcut_id, _timestamp, _options) -> None:  # type: ignore[no-untyped-def]
            if str(session_handle) == self._session_handle and str(shortcut_id) == self._config.shortcut_id:
                on_activate()

        def deactivated(session_handle, shortcut_id, _timestamp, _options) -> None:  # type: ignore[no-untyped-def]
            if str(session_handle) == self._session_handle and str(shortcut_id) == self._config.shortcut_id:
                on_deactivate()

        self._bus.add_signal_receiver(
            activated,
            signal_name="Activated",
            dbus_interface=GLOBAL_SHORTCUTS_IFACE,
        )
        self._bus.add_signal_receiver(
            deactivated,
            signal_name="Deactivated",
            dbus_interface=GLOBAL_SHORTCUTS_IFACE,
        )
        try:
            loop.run()
        finally:
            self.close()

    def close(self) -> None:
        if self._session_handle is None:
            return
        try:
            session = self._bus.get_object(PORTAL_BUS_NAME, self._session_handle)
            iface = self._dbus.Interface(session, "org.freedesktop.portal.Session")
            iface.Close()
        finally:
            self._session_handle = None

    def _open_session(self) -> None:
        create = self._request(
            self._shortcuts.CreateSession,
            {
                "handle_token": self._token("create"),
                "session_handle_token": self._token("session"),
            },
        )
        self._session_handle = str(create["session_handle"])
        self._request(
            self._shortcuts.BindShortcuts,
            self._session_handle,
            self._shortcut_array(),
            "",
            {"handle_token": self._token("bind")},
        )

    def _shortcut_array(self):  # type: ignore[no-untyped-def]
        return self._dbus.Array(
            [
                self._dbus.Struct(
                    (
                        self._dbus.String(self._config.shortcut_id),
                        self._dbus.Dictionary(
                            {
                                "description": self._dbus.String(self._config.description),
                                "preferred_trigger": self._dbus.String(
                                    self._config.preferred_trigger
                                ),
                            },
                            signature="sv",
                        ),
                    ),
                    signature=None,
                )
            ],
            signature="(sa{sv})",
        )

    def _request(self, method, *args):  # type: ignore[no-untyped-def]
        from gi.repository import GLib

        loop = GLib.MainLoop()
        response: dict[str, object] = {}
        error: list[BaseException] = []

        def on_response(code, results) -> None:  # type: ignore[no-untyped-def]
            if int(code) != 0:
                error.append(RuntimeError(f"portal request failed with response code {int(code)}"))
            else:
                response.update(dict(results))
            loop.quit()

        handle = method(*self._variant_args(args))
        self._bus.add_signal_receiver(
            on_response,
            signal_name="Response",
            dbus_interface="org.freedesktop.portal.Request",
            path=str(handle),
        )
        loop.run()
        if error:
            raise error[0]
        return response

    def _variant_args(self, args: tuple[object, ...]) -> tuple[object, ...]:
        converted: list[object] = []
        for arg in args:
            if isinstance(arg, dict):
                converted.append(self._dbus.Dictionary(arg, signature="sv"))
            else:
                converted.append(arg)
        return tuple(converted)

    def _token(self, prefix: str) -> str:
        raw = f"wordpipe_{prefix}_{next(self._tokens)}"
        return re.sub(r"[^A-Za-z0-9_]", "_", raw)
