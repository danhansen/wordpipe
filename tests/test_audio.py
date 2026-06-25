from __future__ import annotations

import sys
import types
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from wordpipe.audio import (
    cpal_input_device_arg,
    list_parakeet_input_devices,
    list_input_devices,
    parse_audio_device,
    render_parakeet_input_devices,
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

    def test_cpal_input_device_arg_accepts_explicit_cpal_index(self) -> None:
        self.assertEqual(cpal_input_device_arg("cpal:2"), "2")

    def test_cpal_input_device_arg_rejects_invalid_explicit_cpal_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid CPAL input device selector"):
            cpal_input_device_arg("cpal:mic")

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

    def test_list_parakeet_input_devices_uses_worker_json_events(self) -> None:
        output = "\n".join(
            [
                '{"event":"input_device","data":{"index":0,"name":"Built-in","is_default":true}}',
                '{"event":"input_device","data":{"index":1,"name":"USB Mic","is_default":false}}',
            ]
        )

        with (
            mock.patch("wordpipe.daemon._resolve_parakeet_worker", return_value=Path("/tmp/worker")),
            mock.patch("wordpipe.daemon.parakeet_worker_env", return_value={}),
            mock.patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["/tmp/worker", "--list-input-devices"],
                    0,
                    stdout=output,
                    stderr="",
                ),
            ) as run,
        ):
            devices = list_parakeet_input_devices(Path("/tmp/worker"))

        run.assert_called_once()
        self.assertEqual([device.selector for device in devices], ["cpal:0", "cpal:1"])
        self.assertEqual([device.name for device in devices], ["Built-in", "USB Mic"])
        self.assertTrue(devices[0].is_default)

    def test_render_parakeet_input_devices_formats_selectors(self) -> None:
        with mock.patch(
            "wordpipe.audio.list_parakeet_input_devices",
            return_value=[
                types.SimpleNamespace(selector="cpal:0", name="Built-in", is_default=True),
                types.SimpleNamespace(selector="cpal:1", name="USB Mic", is_default=False),
            ],
        ):
            rendered = render_parakeet_input_devices()

        self.assertIn("Input devices (Parakeet/CPAL):", rendered)
        self.assertIn("*   cpal:0 Built-in", rendered)
        self.assertIn("    cpal:1 USB Mic", rendered)

    def test_list_parakeet_input_devices_reports_worker_failure(self) -> None:
        with (
            mock.patch("wordpipe.daemon._resolve_parakeet_worker", return_value=Path("/tmp/worker")),
            mock.patch("wordpipe.daemon.parakeet_worker_env", return_value={}),
            mock.patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["/tmp/worker", "--list-input-devices"],
                    2,
                    stdout="",
                    stderr="no audio host",
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "no audio host"),
        ):
            list_parakeet_input_devices(Path("/tmp/worker"))


if __name__ == "__main__":
    unittest.main()
