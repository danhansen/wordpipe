from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any

from .hotkeys import HotkeyMode
from .models import DEFAULT_NEMO_SOURCE_REPO, default_model_root


DEFAULT_CONFIG = """# Wordpipe configuration
# model_dir overrides model_profile/model_root when set.
model_dir = ""
model_profile = "fast"
model_root = ""
nemo_source = ""
asr_runtime = "parakeet"
asr_worker_path = ""
provider = "cpu"
num_threads = 2
sample_rate = 16000
input_device = ""
partial_interval_seconds = 0.10
audio_chunk_seconds = 0.03
queue_seconds = 10.0
stats_interval_seconds = 1.0
enable_endpoint_detection = false
endpoint_rule1_min_trailing_silence = 0.55
endpoint_rule2_min_trailing_silence = 0.35
endpoint_rule3_min_utterance_length = 20.0
overlay = "gtk"
mode = "toggle"
shortcut = "CTRL+ALT+space"
spoken_punctuation = true
dry_run_insertion = false
log_metrics = false
insert_partial_text = false
"""


@dataclass(frozen=True)
class WordpipeConfig:
    model_dir: Path | None = None
    model_profile: str = "fast"
    model_root: Path | None = field(default_factory=default_model_root)
    nemo_source: str = DEFAULT_NEMO_SOURCE_REPO
    asr_runtime: str = "parakeet"
    asr_worker_path: Path | None = None
    provider: str = "cpu"
    num_threads: int = 2
    sample_rate: int = 16000
    input_device: int | str | None = None
    partial_interval_seconds: float = 0.10
    audio_chunk_seconds: float = 0.03
    queue_seconds: float = 10.0
    stats_interval_seconds: float = 1.0
    enable_endpoint_detection: bool = False
    endpoint_rule1_min_trailing_silence: float = 0.55
    endpoint_rule2_min_trailing_silence: float = 0.35
    endpoint_rule3_min_utterance_length: float = 20.0
    overlay: str = "gtk"
    mode: HotkeyMode = "toggle"
    shortcut: str = "CTRL+ALT+space"
    spoken_punctuation: bool = True
    dry_run_insertion: bool = False
    log_metrics: bool = False
    insert_partial_text: bool = False


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "wordpipe" / "config.toml"
    return Path.home() / ".config" / "wordpipe" / "config.toml"


def load_config(path: Path | None = None) -> WordpipeConfig:
    config_path = path if path is not None else default_config_path()
    if not config_path.exists():
        return WordpipeConfig()

    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)

        return WordpipeConfig(
            model_dir=_optional_path(data.get("model_dir")),
            model_profile=_model_profile(data.get("model_profile", "fast")),
            model_root=_optional_path(data.get("model_root")) or default_model_root(),
            nemo_source=_string(data, "nemo_source", DEFAULT_NEMO_SOURCE_REPO)
            or DEFAULT_NEMO_SOURCE_REPO,
            asr_runtime=_runtime(data.get("asr_runtime", "parakeet")),
            asr_worker_path=_optional_path(data.get("asr_worker_path")),
            provider=_string(data, "provider", "cpu"),
            num_threads=_integer(data, "num_threads", 2),
            sample_rate=_integer(data, "sample_rate", 16000),
            input_device=_optional_device(data.get("input_device")),
            partial_interval_seconds=_number(data, "partial_interval_seconds", 0.10),
            audio_chunk_seconds=_number(data, "audio_chunk_seconds", 0.03),
            queue_seconds=_number(data, "queue_seconds", 10.0),
            stats_interval_seconds=_number(data, "stats_interval_seconds", 1.0),
            enable_endpoint_detection=_boolean(data, "enable_endpoint_detection", False),
            endpoint_rule1_min_trailing_silence=_number(
                data, "endpoint_rule1_min_trailing_silence", 0.55
            ),
            endpoint_rule2_min_trailing_silence=_number(
                data, "endpoint_rule2_min_trailing_silence", 0.35
            ),
            endpoint_rule3_min_utterance_length=_number(
                data, "endpoint_rule3_min_utterance_length", 20.0
            ),
            overlay=_string(data, "overlay", "gtk"),
            mode=_mode(data.get("mode", "toggle")),
            shortcut=_string(data, "shortcut", "CTRL+ALT+space"),
            spoken_punctuation=_boolean(data, "spoken_punctuation", True),
            dry_run_insertion=_boolean(data, "dry_run_insertion", False),
            log_metrics=_boolean(data, "log_metrics", False),
            insert_partial_text=_boolean(data, "insert_partial_text", False),
        )
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid config {config_path}: {exc}") from exc


def save_model_profile(profile: str, path: Path | None = None) -> Path:
    selected = _model_profile(profile)
    config_path = path if path is not None else default_config_path()
    existing = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    line = f'model_profile = "{selected}"'
    lines = existing.splitlines()
    replaced = False
    for index, current in enumerate(lines):
        if current.lstrip().split("=", 1)[0].strip() == "model_profile":
            lines[index] = line
            replaced = True
            break
    if not replaced:
        lines.append(line)
    text = "\n".join(lines) + "\n"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=config_path.parent,
        prefix=f"{config_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(config_path)
    return config_path


def _optional_path(value: object) -> Path | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("model_dir must be a string")
    return Path(value).expanduser()


def _runtime(value: object) -> str:
    runtime = _string({"asr_runtime": value}, "asr_runtime", "parakeet")
    if runtime not in {"parakeet", "sherpa"}:
        raise ValueError("asr_runtime must be 'parakeet' or 'sherpa'")
    return runtime


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


def _model_profile(value: object) -> str:
    profile = _string({"model_profile": value}, "model_profile", "fast")
    if profile not in {"fast", "compact"}:
        raise ValueError("model_profile must be 'fast' or 'compact'")
    return profile
