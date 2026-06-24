from __future__ import annotations

import unittest
from unittest import mock

from wordpipe import probe
from wordpipe.probe import (
    GLOBAL_SHORTCUTS_IFACE,
    REMOTE_DESKTOP_IFACE,
    PortalInterface,
    ProbeResult,
)


def _probe_result(*, session_type: str | None = "wayland", gi: bool = True) -> ProbeResult:
    return ProbeResult(
        session_type=session_type,
        gnome_shell="GNOME Shell 50",
        commands={},
        python_modules={"gi": gi},
        portals={
            REMOTE_DESKTOP_IFACE: PortalInterface(REMOTE_DESKTOP_IFACE, True),
            GLOBAL_SHORTCUTS_IFACE: PortalInterface(GLOBAL_SHORTCUTS_IFACE, True),
        },
    )


class ProbeTests(unittest.TestCase):
    def test_probe_is_usable_on_wayland_with_gi_and_required_portals(self) -> None:
        self.assertTrue(_probe_result().usable)

    def test_probe_is_not_usable_outside_wayland(self) -> None:
        self.assertFalse(_probe_result(session_type="x11").usable)
        self.assertFalse(_probe_result(session_type=None).usable)

    def test_probe_is_not_usable_without_gi(self) -> None:
        self.assertFalse(_probe_result(gi=False).usable)

    def test_run_probe_reports_non_wayland_and_missing_gi_errors(self) -> None:
        with (
            mock.patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}),
            mock.patch("wordpipe.probe.shutil.which", return_value=None),
            mock.patch("wordpipe.probe._module_available", return_value=False),
            mock.patch(
                "wordpipe.probe._introspect_portal",
                side_effect=lambda name: PortalInterface(name, True),
            ),
        ):
            result = probe.run_probe()

        self.assertFalse(result.usable)
        self.assertIn("Wayland session", result.errors[0])
        self.assertIn("PyGObject gi", result.errors[1])


if __name__ == "__main__":
    unittest.main()
