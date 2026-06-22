from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any

from .hotkeys import HotkeyMode


DEFAULT_CONFIG = """# Wordpipe configuration
model_dir = "/path/to/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8"
provider = "cpu"
num_threads = 2
sample_rate = 16000
input_device = ""
partial_interval_seconds = 0.10
audio_chunk_seconds = 0.03
stats_interval_seconds = 1.0
endpoint_rule1_min_trailing_silence = 0.55
endpoint_rule2_min_trailing_silence = 0.35
endpoint_rule3_min_utterance_length = 20.0
overlay = "gtk"
mode = "hold"
shortcut = "CTRL+ALT+space"
spoken_punctuation = true
dry_run_insertion = false
log_metrics = false
"""


@dataclass(frozen=True)
class WordpipeConfig:
    model_dir: Path | None = None
    provider: str = "cpu"
    num_threads: int = 2
    sample_rate: int = 16000
    input_device: int | str | None = None
    partial_interval_seconds: float = 0.10
    audio_chunk_seconds: float = 0.03
    stats_interval_seconds: float = 1.0
    endpoint_rule1_min_trailing_silence: float = 0.55
    endpoint_rule2_min_trailing_silence: float = 0.35
    endpoint_rule3_min_utterance_length: float = 20.0
    overlay: str = "stderr"
    mode: HotkeyMode = "hold"
    shortcut: str = "CTRL+ALT+space"
    spoken_punctuation: bool = True
    dry_run_insertion: bool = False
    log_metrics: bool = False


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "wordpipe" / "config.toml"
    return Path.home() / ".config" / "wordpipe" / "config.toml"


def load_config(path: Path | None = None) -> WordpipeConfig:
    config_path = path if path is not None else default_config_path()
    if not config_path.exists():
        return WordpipeConfig()

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    return WordpipeConfig(
        model_dir=_optional_path(data.get("model_dir")),
        provider=_string(data, "provider", "cpu"),
        num_threads=_integer(data, "num_threads", 2),
        sample_rate=_integer(data, "sample_rate", 16000),
        input_device=_optional_device(data.get("input_device")),
        partial_interval_seconds=_number(data, "partial_interval_seconds", 0.10),
        audio_chunk_seconds=_number(data, "audio_chunk_seconds", 0.03),
        stats_interval_seconds=_number(data, "stats_interval_seconds", 1.0),
        endpoint_rule1_min_trailing_silence=_number(
            data, "endpoint_rule1_min_trailing_silence", 0.55
        ),
        endpoint_rule2_min_trailing_silence=_number(
            data, "endpoint_rule2_min_trailing_silence", 0.35
        ),
        endpoint_rule3_min_utterance_length=_number(
            data, "endpoint_rule3_min_utterance_length", 20.0
        ),
        overlay=_string(data, "overlay", "stderr"),
        mode=_mode(data.get("mode", "hold")),
        shortcut=_string(data, "shortcut", "CTRL+ALT+space"),
        spoken_punctuation=_boolean(data, "spoken_punctuation", True),
        dry_run_insertion=_boolean(data, "dry_run_insertion", False),
        log_metrics=_boolean(data, "log_metrics", False),
    )


def _optional_path(value: object) -> Path | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("model_dir must be a string")
    return Path(value).expanduser()


def _optional_device(value: object) -> int | str | None:
    if value is None or value == "":
        return None
    if isinstance(value, int | str):
        return value
    raise ValueError("input_device must be an integer index or string name")


def _string(data: dict[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _integer(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _boolean(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _mode(value: object) -> HotkeyMode:
    if value not in {"hold", "toggle"}:
        raise ValueError("mode must be 'hold' or 'toggle'")
    return value  # type: ignore[return-value]
