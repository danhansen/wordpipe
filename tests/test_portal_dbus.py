from __future__ import annotations

import types
import unittest

from wordpipe.portal_dbus import GioPortalProxy


class FakeLoop:
    def __init__(self) -> None:
        self.quit_called = False

    def run(self) -> None:
        return

    def quit(self) -> None:
        self.quit_called = True


class FakeBus:
    def __init__(self) -> None:
        self.unsubscribed: list[int] = []

    def signal_subscribe(self, *_args):  # type: ignore[no-untyped-def]
        return 42

    def signal_unsubscribe(self, subscription: int) -> None:
        self.unsubscribed.append(subscription)


class PortalDbusTests(unittest.TestCase):
    def test_request_times_out_and_unsubscribes_response_signal(self) -> None:
        loop = FakeLoop()
        bus = FakeBus()
        removed_sources: list[int] = []

        def timeout_add_seconds(_seconds, callback):  # type: ignore[no-untyped-def]
            callback()
            return 7

        proxy = GioPortalProxy.__new__(GioPortalProxy)
        proxy.GLib = types.SimpleNamespace(
            MainLoop=lambda: loop,
            timeout_add_seconds=timeout_add_seconds,
            source_remove=lambda source_id: removed_sources.append(source_id),
            GError=RuntimeError,
        )
        proxy.Gio = types.SimpleNamespace(DBusSignalFlags=types.SimpleNamespace(NONE=0))
        proxy._bus = bus
        proxy.call = lambda *_args: ("/request/wordpipe",)  # type: ignore[method-assign]

        with self.assertRaisesRegex(TimeoutError, "portal request CreateSession timed out"):
            proxy.request("CreateSession", "(a{sv})", ({},), timeout_seconds=1)

        self.assertTrue(loop.quit_called)
        self.assertEqual(bus.unsubscribed, [42])
        self.assertEqual(removed_sources, [])


if __name__ == "__main__":
    unittest.main()
