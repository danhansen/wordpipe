from __future__ import annotations

import unittest

from wordpipe.audio import parse_audio_device


class AudioTests(unittest.TestCase):
    def test_parse_audio_device_none(self) -> None:
        self.assertIsNone(parse_audio_device(None))
        self.assertIsNone(parse_audio_device(""))

    def test_parse_audio_device_index(self) -> None:
        self.assertEqual(parse_audio_device("12"), 12)

    def test_parse_audio_device_name(self) -> None:
        self.assertEqual(parse_audio_device("pipewire"), "pipewire")


if __name__ == "__main__":
    unittest.main()
