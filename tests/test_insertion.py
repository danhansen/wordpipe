from __future__ import annotations

import unittest

from wordpipe.insertion import (
    KEY_PRESSED,
    KEY_RELEASED,
    XK_RETURN,
    DryRunKeyboardBackend,
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

    def test_rejects_non_ascii_for_v1(self) -> None:
        with self.assertRaises(ValueError):
            text_to_key_events("cafe\u0301")

    def test_dry_run_backend_records_events(self) -> None:
        backend = DryRunKeyboardBackend()

        backend.open()
        backend.insert_text("x")
        backend.close()

        self.assertEqual(backend.events, ["open", "press 0x78", "release 0x78", "close"])


if __name__ == "__main__":
    unittest.main()
