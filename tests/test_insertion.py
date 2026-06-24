from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from wordpipe.insertion import (
    KEY_PRESSED,
    KEY_RELEASED,
    XK_RETURN,
    DryRunKeyboardBackend,
    PortalKeyboardBackend,
    sanitize_text_for_keysyms,
    text_to_key_events,
)


class InsertionTests(unittest.TestCase):
    def test_ascii_text_maps_to_press_release_keysyms(self) -> None:
        events = text_to_key_events("A.")

        self.assertEqual(
            [(event.keysym, event.state) for event in events],
            [
                (ord("A"), KEY_PRESSED),
                (ord("A"), KEY_RELEASED),
                (ord("."), KEY_PRESSED),
                (ord("."), KEY_RELEASED),
            ],
        )

    def test_newline_maps_to_return(self) -> None:
        events = text_to_key_events("\n")

        self.assertEqual(events[0].keysym, XK_RETURN)

    def test_sanitizes_non_ascii_for_keysyms(self) -> None:
        self.assertEqual(sanitize_text_for_keysyms("caf\u00e9 \u2014 yes\u2026"), "cafe - yes...")
        events = text_to_key_events("caf\u00e9")
        self.assertEqual(events[-2].keysym, ord("e"))

    def test_dry_run_backend_records_events(self) -> None:
        backend = DryRunKeyboardBackend()

        backend.open()
        backend.insert_text("x")
        backend.close()

        self.assertEqual(backend.events, ["open", "press 0x78", "release 0x78", "close"])

    def test_portal_backend_opens_session_before_first_insert(self) -> None:
        portal = Mock()

        with patch("wordpipe.insertion.RemoteDesktopPortalSession", return_value=portal):
            backend = PortalKeyboardBackend()
            backend.open()
            backend.insert_text("x")
            backend.close()

        portal.open.assert_called_once_with()
        portal.notify_keysym.assert_any_call(ord("x"), KEY_PRESSED)
        portal.notify_keysym.assert_any_call(ord("x"), KEY_RELEASED)
        portal.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
