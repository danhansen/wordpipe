from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave
from typing import Any


AudioDevice = int | str


@dataclass(frozen=True)
class InputDeviceInfo:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    default_samplerate: float
    is_default: bool = False


def parse_audio_device(value: str | None) -> AudioDevice | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def cpal_input_device_arg(device: AudioDevice | None) -> str | None:
    if device is None:
        return None
    if isinstance(device, int):
        return sounddevice_input_device_name(device)
    if device.startswith("cpal:"):
        index = device.removeprefix("cpal:")
        if not index.isdecimal():
            raise ValueError(f"invalid CPAL input device selector: {device}")
        return index
    return device


def sounddevice_input_device_name(index: int) -> str:
    for device in list_input_devices():
        if device.index == index:
            return device.name
    raise ValueError(f"input device index not found: {index}")


def list_input_devices() -> list[InputDeviceInfo]:
    import sounddevice as sd

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    default_input = _default_input_device_index(sd.default.device)
    results: list[InputDeviceInfo] = []
    for index, raw in enumerate(devices):
        max_inputs = int(raw.get("max_input_channels", 0))
        if max_inputs <= 0:
            continue
        hostapi_index = int(raw.get("hostapi", -1))
        hostapi = _hostapi_name(hostapis, hostapi_index)
        results.append(
            InputDeviceInfo(
                index=index,
                name=str(raw.get("name", "")),
                hostapi=hostapi,
                max_input_channels=max_inputs,
                default_samplerate=float(raw.get("default_samplerate", 0.0)),
                is_default=index == default_input,
            )
        )
    return results


def render_input_devices() -> str:
    lines = ["Input devices:"]
    devices = list_input_devices()
    if not devices:
        lines.append("  none found")
        return "\n".join(lines)
    for device in devices:
        marker = "*" if device.is_default else " "
        lines.append(
            f"{marker} {device.index:>3} {device.name} "
            f"({device.hostapi}, inputs={device.max_input_channels}, "
            f"default_sr={device.default_samplerate:g})"
        )
    return "\n".join(lines)


def _default_input_device_index(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, list | tuple) and value:
        first = value[0]
        return first if isinstance(first, int) else None
    return None


def _hostapi_name(hostapis: object, index: int) -> str:
    if index < 0:
        return "unknown"
    try:
        return str(hostapis[index]["name"])  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return "unknown"


def describe_input_device(device: AudioDevice | None) -> dict[str, Any]:
    import sounddevice as sd

    try:
        raw = sd.query_devices(device, "input")
    except Exception as exc:  # noqa: BLE001 - diagnostic path should return the error.
        return {"requested": device, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "requested": device,
        "name": raw.get("name"),
        "max_input_channels": raw.get("max_input_channels"),
        "default_samplerate": raw.get("default_samplerate"),
    }


def record_wav(path: Path, duration_seconds: float, sample_rate: int, device: AudioDevice | None) -> None:
    import numpy as np
    import sounddevice as sd

    frames = max(1, int(duration_seconds * sample_rate))
    samples = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    pcm = np.clip(samples[:, 0], -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
