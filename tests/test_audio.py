from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from wordpipe.audio import (
    cpal_input_device_arg,
    list_input_devices,
    parse_audio_device,
    render_input_devices,
)


class AudioTests(unittest.TestCase):
    def test_parse_audio_device_none(self) -> None:
        self.assertIsNone(parse_audio_device(None))
        self.assertIsNone(parse_audio_device(""))

    def test_parse_audio_device_index(self) -> None:
        self.assertEqual(parse_audio_device("12"), 12)

    def test_parse_audio_device_name(self) -> None:
        self.assertEqual(parse_audio_device("pipewire"), "pipewire")

    def test_cpal_input_device_arg_resolves_sounddevice_index_to_name(self) -> None:
        sounddevice = types.SimpleNamespace(
            default=types.SimpleNamespace(device=(None, None)),
            query_devices=lambda: [
                {"name": "output only", "hostapi": 0, "max_input_channels": 0},
                {
                    "name": "Built-in Microphone",
                    "hostapi": 0,
                    "max_input_channels": 1,
                    "default_samplerate": 48000,
                },
            ],
            query_hostapis=lambda: [{"name": "PipeWire"}],
        )

        with mock.patch.dict(sys.modules, {"sounddevice": sounddevice}):
            self.assertEqual(cpal_input_device_arg(1), "Built-in Microphone")

    def test_cpal_input_device_arg_preserves_name(self) -> None:
        self.assertEqual(cpal_input_device_arg("Built-in"), "Built-in")

    def test_cpal_input_device_arg_rejects_unknown_index(self) -> None:
        sounddevice = types.SimpleNamespace(
            default=types.SimpleNamespace(device=(None, None)),
            query_devices=lambda: [],
            query_hostapis=lambda: [],
        )

        with mock.patch.dict(sys.modules, {"sounddevice": sounddevice}):
            with self.assertRaisesRegex(ValueError, "input device index not found: 3"):
                cpal_input_device_arg(3)

    def test_list_input_devices_accepts_integer_default_device(self) -> None:
        sounddevice = types.SimpleNamespace(
            default=types.SimpleNamespace(device=1),
            query_devices=lambda: [
                {"name": "output only", "hostapi": 0, "max_input_channels": 0},
                {
                    "name": "mic",
                    "hostapi": 0,
                    "max_input_channels": 1,
                    "default_samplerate": 48000,
                },
            ],
            query_hostapis=lambda: [{"name": "PipeWire"}],
        )

        with mock.patch.dict(sys.modules, {"sounddevice": sounddevice}):
            devices = list_input_devices()

        self.assertEqual(len(devices), 1)
        self.assertTrue(devices[0].is_default)
        self.assertEqual(devices[0].hostapi, "PipeWire")

    def test_list_input_devices_handles_unknown_hostapi_index(self) -> None:
        sounddevice = types.SimpleNamespace(
            default=types.SimpleNamespace(device=(0, None)),
            query_devices=lambda: [
                {
                    "name": "mic",
                    "hostapi": 99,
                    "max_input_channels": 1,
                    "default_samplerate": 16000,
                },
            ],
            query_hostapis=lambda: [{"name": "PipeWire"}],
        )

        with mock.patch.dict(sys.modules, {"sounddevice": sounddevice}):
            devices = list_input_devices()

        self.assertEqual(devices[0].hostapi, "unknown")

    def test_render_input_devices_reports_empty_list(self) -> None:
        sounddevice = types.SimpleNamespace(
            default=types.SimpleNamespace(device=(None, None)),
            query_devices=lambda: [],
            query_hostapis=lambda: [],
        )

        with mock.patch.dict(sys.modules, {"sounddevice": sounddevice}):
            rendered = render_input_devices()

        self.assertIn("none found", rendered)


if __name__ == "__main__":
    unittest.main()
