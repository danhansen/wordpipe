from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import threading

from .audio import AudioDevice
from .daemon import AsrProcess, DaemonConfig


@dataclass(frozen=True)
class ListenTestConfig:
    model_dir: Path
    asr_runtime: str = "parakeet"
    asr_worker_path: Path | None = None
    provider: str = "cpu"
    num_threads: int = 2
    sample_rate: int = 16000
    input_device: AudioDevice | None = None
    partial_interval_seconds: float = 0.05
    audio_chunk_seconds: float = 0.03
    queue_seconds: float = 10.0
    stats_interval_seconds: float = 1.0
    enable_endpoint_detection: bool = False
    endpoint_rule1_min_trailing_silence: float = 0.55
    endpoint_rule2_min_trailing_silence: float = 0.35
    endpoint_rule3_min_utterance_length: float = 20.0
    duration_seconds: float | None = None
    json_output: bool = False
    full_hypotheses: bool = False


def run_listen_test(config: ListenTestConfig) -> int:
    daemon_config = DaemonConfig(
        model_dir=config.model_dir,
        asr_runtime=config.asr_runtime,
        asr_worker_path=config.asr_worker_path,
        provider=config.provider,
        num_threads=config.num_threads,
        sample_rate=config.sample_rate,
        input_device=config.input_device,
        partial_interval_seconds=config.partial_interval_seconds,
        audio_chunk_seconds=config.audio_chunk_seconds,
        queue_seconds=config.queue_seconds,
        stats_interval_seconds=config.stats_interval_seconds,
        enable_endpoint_detection=config.enable_endpoint_detection,
        endpoint_rule1_min_trailing_silence=config.endpoint_rule1_min_trailing_silence,
        endpoint_rule2_min_trailing_silence=config.endpoint_rule2_min_trailing_silence,
        endpoint_rule3_min_utterance_length=config.endpoint_rule3_min_utterance_length,
    )
    asr = AsrProcess(daemon_config)
    timer: threading.Timer | None = None
    formatter = ListenTestFormatter(full_hypotheses=config.full_hypotheses)

    def start_timer_once() -> None:
        nonlocal timer
        if config.duration_seconds is None or timer is not None:
            return
        timer = threading.Timer(config.duration_seconds, asr.send, args=("stop",))
        timer.daemon = True
        timer.start()

    asr.start()
    try:
        asr.send("start")
        for item in asr.events():
            if config.json_output:
                print(json.dumps(item, sort_keys=True), flush=True)
            else:
                print(formatter.format(item), flush=True)
            if item.get("event") == "listening":
                start_timer_once()
            if item.get("event") == "error":
                return 1
            if item.get("event") == "stopped":
                return 0
    finally:
        if timer is not None:
            timer.cancel()
        asr.close()
    return 0


class ListenTestFormatter:
    def __init__(self, full_hypotheses: bool = False) -> None:
        self._full_hypotheses = full_hypotheses
        self._last_text = ""

    def format(self, item: dict[str, object]) -> str:
        return _format_event_with_state(
            item,
            previous_text=self._last_text,
            full_hypotheses=self._full_hypotheses,
            remember_text=self._remember_text,
        )

    def _remember_text(self, text: str) -> None:
        if text:
            self._last_text = text


def _format_event(item: dict[str, object]) -> str:
    return _format_event_with_state(item, previous_text="", full_hypotheses=True)


def _format_event_with_state(
    item: dict[str, object],
    *,
    previous_text: str,
    full_hypotheses: bool,
    remember_text=None,  # type: ignore[no-untyped-def]
) -> str:
    kind = str(item.get("event", "unknown"))
    text = str(item.get("text", ""))
    metrics = _format_metrics(item.get("data"))
    if kind in {"partial", "commit"}:
        display = text if full_hypotheses else _new_suffix(previous_text, text)
        if remember_text is not None:
            remember_text(text)
        label = kind if full_hypotheses else f"{kind}+"
        return f"{label:7} {metrics} {display}".rstrip()
    if kind == "stats":
        line = f"stats   {metrics}"
        if text:
            display = text if full_hypotheses else _new_suffix(previous_text, text)
            if display:
                label = "partial" if full_hypotheses else "partial+"
                line = f"{line}\n{label:7} {metrics} {display}"
            if remember_text is not None:
                remember_text(text)
        return line
    if kind == "listening":
        return f"listening {_format_device(item.get('data'))}"
    if kind == "error":
        return f"error   {item.get('message', '')}"
    return kind


def _new_suffix(previous: str, current: str) -> str:
    if not previous:
        return current
    if current.startswith(previous):
        return current[len(previous) :].lstrip()
    return current


def _format_device(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    device = data.get("input_device")
    if not isinstance(device, dict):
        return ""
    if "error" in device:
        return f"input_device_error={device.get('error')}"
    return f"input_device={device.get('name')} requested={device.get('requested')}"


def _format_metrics(data: object) -> str:
    if not isinstance(data, dict):
        return "rtf=?"
    rtf = data.get("real_time_factor", "?")
    audio = data.get("audio_seconds", "?")
    elapsed = data.get("elapsed_seconds", "?")
    decode = data.get("decode_seconds", "?")
    dropped = data.get("dropped_audio_chunks", "?")
    rms = data.get("last_rms", "?")
    peak = data.get("peak_rms", "?")
    return (
        f"rtf={rtf} audio={audio}s decode={decode}s elapsed={elapsed}s "
        f"rms={rms} peak={peak} dropped={dropped}"
    )
