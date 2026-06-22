from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Iterator

from .hotkeys import (
    GlobalShortcutsPortalLoop,
    HotkeyConfig,
    HotkeyLoop,
    HotkeyMode,
    HotkeyStateMachine,
    ManualHotkeyLoop,
)
from .audio import AudioDevice
from .insertion import DryRunKeyboardBackend, KeyboardBackend, PortalKeyboardBackend
from .normalization import normalize_spoken_punctuation
from .transcript import StderrTranscriptSink, TranscriptSink


@dataclass(frozen=True)
class DaemonConfig:
    model_dir: Path
    dry_run_insertion: bool = False
    provider: str = "cpu"
    num_threads: int = 2
    sample_rate: int = 16000
    input_device: AudioDevice | None = None
    partial_interval_seconds: float = 0.10
    audio_chunk_seconds: float = 0.03
    queue_seconds: float = 10.0
    stats_interval_seconds: float = 1.0
    endpoint_rule1_min_trailing_silence: float = 0.55
    endpoint_rule2_min_trailing_silence: float = 0.35
    endpoint_rule3_min_utterance_length: float = 20.0
    spoken_punctuation: bool = True
    log_metrics: bool = False


def format_committed_text(text: str) -> str:
    stripped = text.strip(" \t")
    if not stripped:
        return ""
    if stripped.endswith("\n"):
        return stripped
    return f"{stripped} "


class AsrProcess:
    def __init__(self, config: DaemonConfig) -> None:
        self._config = config
        self._proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._proc is not None:
            return

        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1])
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = source_root if not existing else f"{source_root}{os.pathsep}{existing}"

        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "wordpipe",
                "asr-worker",
                "--model-dir",
                str(self._config.model_dir),
                "--provider",
                self._config.provider,
                "--num-threads",
                str(self._config.num_threads),
                "--sample-rate",
                str(self._config.sample_rate),
                *(
                    ["--input-device", str(self._config.input_device)]
                    if self._config.input_device is not None
                    else []
                ),
                "--partial-interval-seconds",
                str(self._config.partial_interval_seconds),
                "--audio-chunk-seconds",
                str(self._config.audio_chunk_seconds),
                "--queue-seconds",
                str(self._config.queue_seconds),
                "--stats-interval-seconds",
                str(self._config.stats_interval_seconds),
                "--endpoint-rule1-min-trailing-silence",
                str(self._config.endpoint_rule1_min_trailing_silence),
                "--endpoint-rule2-min-trailing-silence",
                str(self._config.endpoint_rule2_min_trailing_silence),
                "--endpoint-rule3-min-utterance-length",
                str(self._config.endpoint_rule3_min_utterance_length),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    def send(self, command: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("ASR process is not running")
        self._proc.stdin.write(json.dumps({"command": command}) + "\n")
        self._proc.stdin.flush()

    def events(self) -> Iterator[dict[str, object]]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("ASR process is not running")
        for line in self._proc.stdout:
            if not line.strip():
                continue
            yield json.loads(line)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self.send("shutdown")
                self._proc.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        finally:
            self._proc = None


class DictationDaemon:
    def __init__(
        self,
        config: DaemonConfig,
        keyboard: KeyboardBackend,
        transcript: TranscriptSink | None = None,
    ) -> None:
        self._controller = DictationController(config, keyboard, transcript)

    def run(self) -> int:
        self._controller.open()
        try:
            self._controller.start_dictation()
            return self._controller.wait()
        finally:
            self._controller.close()


class DictationController:
    def __init__(
        self,
        config: DaemonConfig,
        keyboard: KeyboardBackend,
        transcript: TranscriptSink | None = None,
    ) -> None:
        self._config = config
        self._keyboard = keyboard
        self._transcript = transcript if transcript is not None else StderrTranscriptSink()
        self._asr = AsrProcess(config)
        self._reader: threading.Thread | None = None
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._opened = False
        self._listening = False
        self._exit_code = 0

    @property
    def listening(self) -> bool:
        with self._lock:
            return self._listening

    def open(self) -> None:
        if self._opened:
            return
        self._transcript.open()
        self._keyboard.open()
        if isinstance(self._keyboard, DryRunKeyboardBackend):
            self._keyboard.events.clear()
        self._asr.start()
        self._reader = threading.Thread(target=self._read_events, name="wordpipe-events", daemon=True)
        self._reader.start()
        self._opened = True

    def start_dictation(self) -> None:
        self.open()
        with self._lock:
            if self._listening:
                return
            self._listening = True
        self._transcript.status("starting dictation")
        self._asr.send("start")

    def stop_dictation(self) -> None:
        with self._lock:
            if not self._listening:
                return
            self._listening = False
        self._transcript.status("stopping dictation")
        self._asr.send("stop")

    def wait(self) -> int:
        self._done.wait()
        return self._exit_code

    def close(self) -> None:
        try:
            self._asr.close()
        finally:
            self._keyboard.close()
            self._transcript.close()
            with self._lock:
                self._listening = False
            self._done.set()

    def _read_events(self) -> None:
        try:
            for item in self._asr.events():
                self._handle_event(item)
        except Exception as exc:  # noqa: BLE001 - daemon must surface subprocess failures.
            self._transcript.error(f"{type(exc).__name__}: {exc}")
            self._exit_code = 1
            self._done.set()

    def _handle_event(self, item: dict[str, object]) -> None:
        kind = item.get("event")
        if kind == "ready":
            self._transcript.status("ASR worker ready")
        elif kind == "listening":
            self._transcript.status("listening")
        elif kind == "partial":
            self._transcript.partial(str(item.get("text", "")))
        elif kind == "commit":
            raw_text = str(item.get("text", ""))
            if self._config.spoken_punctuation:
                raw_text = normalize_spoken_punctuation(raw_text)
            text = format_committed_text(raw_text)
            if text:
                self._transcript.commit(text)
                if self._config.log_metrics:
                    self._transcript.status(_format_metrics(item.get("data")))
                self._keyboard.insert_text(text)
                if isinstance(self._keyboard, DryRunKeyboardBackend):
                    for rendered in self._keyboard.events:
                        print(f"key: {rendered}", file=sys.stderr)
                    self._keyboard.events.clear()
        elif kind == "error":
            self._transcript.error(str(item.get("message", "")))
            self._exit_code = 1
            self._done.set()
        elif kind == "stopped":
            with self._lock:
                self._listening = False


def _format_metrics(data: object) -> str:
    if not isinstance(data, dict):
        return "metrics: unavailable"
    rtf = data.get("real_time_factor", "?")
    audio = data.get("audio_seconds", "?")
    elapsed = data.get("elapsed_seconds", "?")
    dropped = data.get("dropped_audio_chunks", "?")
    return f"metrics: rtf={rtf} audio={audio}s elapsed={elapsed}s dropped={dropped}"


def run_daemon(config: DaemonConfig, transcript: TranscriptSink | None = None) -> int:
    keyboard: KeyboardBackend
    keyboard = DryRunKeyboardBackend() if config.dry_run_insertion else PortalKeyboardBackend()
    daemon = DictationDaemon(config, keyboard, transcript)
    return daemon.run()


def run_hotkey_daemon(
    config: DaemonConfig,
    mode: HotkeyMode,
    shortcut: str,
    manual_hotkey: bool = False,
    transcript: TranscriptSink | None = None,
) -> int:
    keyboard: KeyboardBackend
    keyboard = DryRunKeyboardBackend() if config.dry_run_insertion else PortalKeyboardBackend()
    controller = DictationController(config, keyboard, transcript)
    hotkey_loop: HotkeyLoop
    if manual_hotkey:
        hotkey_loop = ManualHotkeyLoop(sys.stdin, sys.stderr)
    else:
        hotkey_loop = GlobalShortcutsPortalLoop(
            HotkeyConfig(mode=mode, preferred_trigger=shortcut)
        )
    state = HotkeyStateMachine(
        mode=mode,
        start=controller.start_dictation,
        stop=controller.stop_dictation,
        is_listening=lambda: controller.listening,
    )
    controller.open()
    try:
        hotkey_loop.run(state.activate, state.deactivate)
        return 0
    finally:
        controller.close()
