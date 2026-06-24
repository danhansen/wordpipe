from __future__ import annotations

from dataclasses import dataclass
import itertools
import re
import unicodedata
from typing import Protocol

from .probe import REMOTE_DESKTOP_IFACE
from .portal_dbus import GioPortalProxy, variant_options, variant_uint32


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
    for char in sanitize_text_for_keysyms(text):
        keysym = _char_to_keysym(char)
        events.append(KeyEvent(keysym, KEY_PRESSED))
        events.append(KeyEvent(keysym, KEY_RELEASED))
    return events


def sanitize_text_for_keysyms(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    normalized = unicodedata.normalize("NFKD", text)
    output: list[str] = []
    for char in normalized:
        char = replacements.get(char, char)
        if char in {"\n", "\t"} or 0x20 <= ord(char) <= 0x7E:
            output.append(char)
        elif unicodedata.category(char).startswith("M"):
            continue
        else:
            output.append(" ")
    return re.sub(r"[ \t]{2,}", " ", "".join(output))


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
        if self._portal is None:
            portal = RemoteDesktopPortalSession()
            portal.open()
            self._portal = portal

    def insert_text(self, text: str) -> None:
        if self._portal is None:
            self.open()
        assert self._portal is not None
        for key_event in text_to_key_events(text):
            self._portal.notify_keysym(key_event.keysym, key_event.state)

    def close(self) -> None:
        if self._portal is not None:
            self._portal.close()
            self._portal = None


class RemoteDesktopPortalSession:
    _tokens = itertools.count(1)

    def __init__(self) -> None:
        self._remote = GioPortalProxy(REMOTE_DESKTOP_IFACE)
        self._session_handle: str | None = None

    def open(self) -> None:
        try:
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
                "SelectDevices",
                "(oa{sv})",
                self._session_handle,
                {
                    "handle_token": self._token("devices"),
                    "types": variant_uint32(KEYBOARD_DEVICE),
                },
            )
            self._request(
                "Start",
                "(osa{sv})",
                self._session_handle,
                "",
                {"handle_token": self._token("start")},
            )
        except Exception:
            self.close()
            raise

    def notify_keysym(self, keysym: int, state: int) -> None:
        if self._session_handle is None:
            raise RuntimeError("remote desktop portal session is not open")
        self._remote.call(
            "NotifyKeyboardKeysym",
            "(oa{sv}iu)",
            (self._session_handle, {}, keysym, state),
        )

    def close(self) -> None:
        if self._session_handle is None:
            return
        try:
            self._remote.close_session(self._session_handle)
        finally:
            self._session_handle = None

    def _request(self, method: str, signature: str, *args: object) -> dict[str, object]:
        return self._remote.request(method, signature, self._variant_args(args))

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
