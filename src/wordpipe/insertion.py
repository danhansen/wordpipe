from __future__ import annotations

from dataclasses import dataclass
import itertools
import re
from typing import Protocol

from .probe import PORTAL_BUS_NAME, PORTAL_OBJECT_PATH, REMOTE_DESKTOP_IFACE


KEYBOARD_DEVICE = 1
KEY_RELEASED = 0
KEY_PRESSED = 1
XK_RETURN = 0xFF0D
XK_TAB = 0xFF09


class KeyboardBackend(Protocol):
    def open(self) -> None:
        ...

    def insert_text(self, text: str) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class KeyEvent:
    keysym: int
    state: int

    def render(self) -> str:
        state_name = "press" if self.state == KEY_PRESSED else "release"
        return f"{state_name} 0x{self.keysym:x}"


def text_to_key_events(text: str) -> list[KeyEvent]:
    events: list[KeyEvent] = []
    for char in text:
        keysym = _char_to_keysym(char)
        events.append(KeyEvent(keysym, KEY_PRESSED))
        events.append(KeyEvent(keysym, KEY_RELEASED))
    return events


def _char_to_keysym(char: str) -> int:
    if char == "\n":
        return XK_RETURN
    if char == "\t":
        return XK_TAB
    codepoint = ord(char)
    if 0x20 <= codepoint <= 0x7E:
        return codepoint
    raise ValueError(f"unsupported character for v1 keyboard insertion: U+{codepoint:04X}")


class DryRunKeyboardBackend:
    def __init__(self) -> None:
        self.events: list[str] = []

    def open(self) -> None:
        self.events.append("open")

    def insert_text(self, text: str) -> None:
        self.events.extend(event.render() for event in text_to_key_events(text))

    def close(self) -> None:
        self.events.append("close")


class PortalKeyboardBackend:
    def __init__(self) -> None:
        self._portal: RemoteDesktopPortalSession | None = None

    def open(self) -> None:
        self._portal = RemoteDesktopPortalSession()
        self._portal.open()

    def insert_text(self, text: str) -> None:
        if self._portal is None:
            raise RuntimeError("keyboard backend is not open")
        for key_event in text_to_key_events(text):
            self._portal.notify_keysym(key_event.keysym, key_event.state)

    def close(self) -> None:
        if self._portal is not None:
            self._portal.close()
            self._portal = None


class RemoteDesktopPortalSession:
    _tokens = itertools.count(1)

    def __init__(self) -> None:
        try:
            import dbus
            from dbus.mainloop.glib import DBusGMainLoop
        except ImportError as exc:
            raise RuntimeError("dbus-python is required for portal keyboard insertion") from exc

        DBusGMainLoop(set_as_default=True)
        self._dbus = dbus
        self._bus = dbus.SessionBus()
        obj = self._bus.get_object(PORTAL_BUS_NAME, PORTAL_OBJECT_PATH)
        self._remote = dbus.Interface(obj, REMOTE_DESKTOP_IFACE)
        self._session_handle: str | None = None

    def open(self) -> None:
        create = self._request(
            self._remote.CreateSession,
            {
                "handle_token": self._token("create"),
                "session_handle_token": self._token("session"),
            },
        )
        self._session_handle = str(create["session_handle"])

        self._request(
            self._remote.SelectDevices,
            self._session_handle,
            {
                "handle_token": self._token("devices"),
                "types": self._dbus.UInt32(KEYBOARD_DEVICE),
            },
        )
        self._request(
            self._remote.Start,
            self._session_handle,
            "",
            {"handle_token": self._token("start")},
        )

    def notify_keysym(self, keysym: int, state: int) -> None:
        if self._session_handle is None:
            raise RuntimeError("remote desktop portal session is not open")
        self._remote.NotifyKeyboardKeysym(
            self._session_handle,
            self._dbus.Dictionary({}, signature="sv"),
            self._dbus.Int32(keysym),
            self._dbus.UInt32(state),
        )

    def close(self) -> None:
        if self._session_handle is None:
            return
        try:
            session = self._bus.get_object(PORTAL_BUS_NAME, self._session_handle)
            iface = self._dbus.Interface(session, "org.freedesktop.portal.Session")
            iface.Close()
        finally:
            self._session_handle = None

    def _request(self, method, *args):  # type: ignore[no-untyped-def]
        from gi.repository import GLib

        loop = GLib.MainLoop()
        response: dict[str, object] = {}
        error: list[BaseException] = []
        expected_handle: list[str] = []

        def on_response(code, results, request_path=None) -> None:  # type: ignore[no-untyped-def]
            if expected_handle and str(request_path) != expected_handle[0]:
                return
            if int(code) != 0:
                error.append(RuntimeError(f"portal request failed with response code {int(code)}"))
            else:
                response.update(dict(results))
            loop.quit()

        match = self._bus.add_signal_receiver(
            on_response,
            signal_name="Response",
            dbus_interface="org.freedesktop.portal.Request",
            path_keyword="request_path",
        )
        try:
            handle = method(*self._variant_args(args))
            expected_handle.append(str(handle))
            loop.run()
            if error:
                raise error[0]
            return response
        finally:
            if match is not None:
                match.remove()

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
