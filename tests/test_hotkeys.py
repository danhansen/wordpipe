from __future__ import annotations

import io
import unittest

from wordpipe.hotkeys import HotkeyStateMachine, ManualHotkeyLoop


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


if __name__ == "__main__":
    unittest.main()
