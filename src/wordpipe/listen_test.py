from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import threading

from .daemon import AsrProcess, DaemonConfig


@dataclass(frozen=True)
class ListenTestConfig:
    model_dir: Path
    provider: str = "cpu"
    num_threads: int = 2
    sample_rate: int = 16000
    partial_interval_seconds: float = 0.05
    audio_chunk_seconds: float = 0.03
    stats_interval_seconds: float = 1.0
    endpoint_rule1_min_trailing_silence: float = 0.55
    endpoint_rule2_min_trailing_silence: float = 0.35
    endpoint_rule3_min_utterance_length: float = 20.0
    duration_seconds: float | None = None
    json_output: bool = False


def run_listen_test(config: ListenTestConfig) -> int:
    daemon_config = DaemonConfig(
        model_dir=config.model_dir,
        provider=config.provider,
        num_threads=config.num_threads,
        sample_rate=config.sample_rate,
        partial_interval_seconds=config.partial_interval_seconds,
        audio_chunk_seconds=config.audio_chunk_seconds,
        stats_interval_seconds=config.stats_interval_seconds,
        endpoint_rule1_min_trailing_silence=config.endpoint_rule1_min_trailing_silence,
        endpoint_rule2_min_trailing_silence=config.endpoint_rule2_min_trailing_silence,
        endpoint_rule3_min_utterance_length=config.endpoint_rule3_min_utterance_length,
    )
    asr = AsrProcess(daemon_config)
    timer: threading.Timer | None = None

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
                print(_format_event(item), flush=True)
            if item.get("event") == "listening":
                start_timer_once()
            if item.get("event") == "stopped":
                return 0
    finally:
        if timer is not None:
            timer.cancel()
        asr.close()
    return 0


def _format_event(item: dict[str, object]) -> str:
    kind = str(item.get("event", "unknown"))
    text = str(item.get("text", ""))
    metrics = _format_metrics(item.get("data"))
    if kind in {"partial", "commit"}:
        return f"{kind:7} {metrics} {text}".rstrip()
    if kind == "stats":
        return f"stats   {metrics}"
    if kind == "error":
        return f"error   {item.get('message', '')}"
    return kind


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
