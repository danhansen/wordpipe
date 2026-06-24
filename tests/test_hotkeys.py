from __future__ import annotations

import io
import sys
import types
import unittest
from unittest import mock

from wordpipe.hotkeys import (
    GlobalShortcutsPortalLoop,
    HotkeyConfig,
    HotkeyStateMachine,
    ManualHotkeyLoop,
)


class HotkeyStateMachineTests(unittest.TestCase):
    def test_hold_mode_starts_on_activate_and_stops_on_deactivate(self) -> None:
        calls: list[str] = []
        state = HotkeyStateMachine(
            "hold",
            start=lambda: calls.append("start"),
            stop=lambda: calls.append("stop"),
            is_listening=lambda: bool(calls and calls[-1] == "start"),
        )

        state.activate()
        state.deactivate()

        self.assertEqual(calls, ["start", "stop"])

    def test_toggle_mode_toggles_on_activation(self) -> None:
        calls: list[str] = []
        listening = False

        def start() -> None:
            nonlocal listening
            listening = True
            calls.append("start")

        def stop() -> None:
            nonlocal listening
            listening = False
            calls.append("stop")

        state = HotkeyStateMachine("toggle", start, stop, lambda: listening)

        state.activate()
        state.deactivate()
        state.activate()

        self.assertEqual(calls, ["start", "stop"])

    def test_manual_loop_maps_down_and_up(self) -> None:
        calls: list[str] = []
        loop = ManualHotkeyLoop(io.StringIO("down\nup\nquit\n"), io.StringIO())

        loop.run(lambda: calls.append("down"), lambda: calls.append("up"))

        self.assertEqual(calls, ["down", "up"])


class GlobalShortcutsPortalLoopTests(unittest.TestCase):
    def test_open_session_closes_created_session_when_bind_fails(self) -> None:
        shortcuts = mock.Mock()
        shortcuts.request.side_effect = [
            {"session_handle": "/session/wordpipe"},
            RuntimeError("bind failed"),
        ]
        loop = GlobalShortcutsPortalLoop.__new__(GlobalShortcutsPortalLoop)
        loop._config = HotkeyConfig()  # type: ignore[attr-defined]
        loop._shortcuts = shortcuts  # type: ignore[attr-defined]
        loop._session_handle = None  # type: ignore[attr-defined]

        with self.assertRaisesRegex(RuntimeError, "bind failed"):
            loop._open_session()

        shortcuts.close_session.assert_called_once_with("/session/wordpipe")
        self.assertIsNone(loop._session_handle)

    def test_run_cleans_partial_signal_subscription_when_second_subscribe_fails(self) -> None:
        shortcuts = mock.Mock()
        shortcuts.subscribe_signal.side_effect = [7, RuntimeError("subscribe failed")]
        loop = GlobalShortcutsPortalLoop.__new__(GlobalShortcutsPortalLoop)
        loop._config = HotkeyConfig()  # type: ignore[attr-defined]
        loop._shortcuts = shortcuts  # type: ignore[attr-defined]
        loop._session_handle = None  # type: ignore[attr-defined]
        def open_session() -> None:
            loop._session_handle = "/session/wordpipe"  # type: ignore[attr-defined]

        loop._open_session = open_session  # type: ignore[method-assign]
        fake_gi = types.ModuleType("gi")
        fake_repository = types.ModuleType("gi.repository")
        fake_glib = types.SimpleNamespace(MainLoop=lambda: types.SimpleNamespace(run=lambda: None))
        fake_repository.GLib = fake_glib

        with (
            mock.patch.dict(sys.modules, {"gi": fake_gi, "gi.repository": fake_repository}),
            self.assertRaisesRegex(RuntimeError, "subscribe failed"),
        ):
            loop.run(lambda: None, lambda: None)

        shortcuts.unsubscribe_signal.assert_called_once_with(7)
        shortcuts.close_session.assert_called_once_with("/session/wordpipe")
        self.assertIsNone(loop._session_handle)


if __name__ == "__main__":
    unittest.main()
