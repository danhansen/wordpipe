from __future__ import annotations

from dataclasses import dataclass
import shutil
import json
import math
import os
from pathlib import Path
import signal
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
from .audio import AudioDevice, cpal_input_device_arg
from .insertion import DryRunKeyboardBackend, KeyboardBackend, PortalKeyboardBackend
from .normalization import normalize_spoken_punctuation
from .transcript import StderrTranscriptSink, TranscriptSink


def _resolve_parakeet_worker(configured_path: Path | None = None) -> Path | str:
    if configured_path is not None:
        return configured_path

    repo_root = Path(__file__).resolve().parents[2]
    release_binary = repo_root / "target" / "release" / "wordpipe-parakeet-worker"
    if release_binary.exists():
        return release_binary

    debug_binary = repo_root / "target" / "debug" / "wordpipe-parakeet-worker"
    if debug_binary.exists():
        return debug_binary

    installed = shutil.which("wordpipe-parakeet-worker")
    if installed is not None:
        return installed

    return "wordpipe-parakeet-worker"


def _default_ort_dylib_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    patterns = (
        "/app/lib/libonnxruntime.so*",
        "/app/lib/onnxruntime/libonnxruntime.so*",
        "/app/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*",
        ".venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*",
        ".venv-nemo-export/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*",
        ".venv/lib/python*/site-packages/sherpa_onnx/lib/libonnxruntime.so*",
    )
    for pattern in patterns:
        if pattern.startswith("/"):
            matches = sorted(Path("/").glob(pattern.lstrip("/")))
        else:
            matches = sorted(repo_root.glob(pattern))
        if matches:
            return matches[0]
    return None


def parakeet_worker_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    if "ORT_DYLIB_PATH" not in env:
        dylib = _default_ort_dylib_path()
        if dylib is not None:
            env["ORT_DYLIB_PATH"] = str(dylib)
    return env


@dataclass(frozen=True)
class DaemonConfig:
    model_dir: Path
    asr_runtime: str = "parakeet"
    asr_worker_path: Path | None = None
    dry_run_insertion: bool = False
    provider: str = "cpu"
    num_threads: int = 2
    sample_rate: int = 16000
    input_device: AudioDevice | None = None
    partial_interval_seconds: float = 0.10
    audio_chunk_seconds: float = 0.03
    queue_seconds: float = 10.0
    stats_interval_seconds: float = 1.0
    enable_endpoint_detection: bool = False
    endpoint_rule1_min_trailing_silence: float = 0.55
    endpoint_rule2_min_trailing_silence: float = 0.35
    endpoint_rule3_min_utterance_length: float = 20.0
    spoken_punctuation: bool = True
    log_metrics: bool = False
    insert_partial_text: bool = False

    def __post_init__(self) -> None:
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        _require_positive_finite(self.partial_interval_seconds, "partial_interval_seconds")
        _require_positive_finite(self.audio_chunk_seconds, "audio_chunk_seconds")
        _require_positive_finite(self.queue_seconds, "queue_seconds")
        _require_positive_finite(self.stats_interval_seconds, "stats_interval_seconds")
        _require_positive_finite(
            self.endpoint_rule1_min_trailing_silence,
            "endpoint_rule1_min_trailing_silence",
        )
        _require_positive_finite(
            self.endpoint_rule2_min_trailing_silence,
            "endpoint_rule2_min_trailing_silence",
        )
        _require_positive_finite(
            self.endpoint_rule3_min_utterance_length,
            "endpoint_rule3_min_utterance_length",
        )


def _require_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive")


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

        if self._config.asr_runtime == "parakeet":
            env = parakeet_worker_env(env)

        self._proc = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    def _command(self) -> list[str]:
        if self._config.asr_runtime == "sherpa":
            return [
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
                *(["--endpoint"] if self._config.enable_endpoint_detection else []),
                "--endpoint-rule1-min-trailing-silence",
                str(self._config.endpoint_rule1_min_trailing_silence),
                "--endpoint-rule2-min-trailing-silence",
                str(self._config.endpoint_rule2_min_trailing_silence),
                "--endpoint-rule3-min-utterance-length",
                str(self._config.endpoint_rule3_min_utterance_length),
            ]

        if self._config.asr_runtime == "parakeet":
            input_device = cpal_input_device_arg(self._config.input_device)
            return [
                str(_resolve_parakeet_worker(self._config.asr_worker_path)),
                "--model-dir",
                str(self._config.model_dir),
                "--num-threads",
                str(self._config.num_threads),
                "--sample-rate",
                str(self._config.sample_rate),
                *(
                    ["--input-device", input_device]
                    if input_device is not None
                    else []
                ),
                "--queue-seconds",
                str(self._config.queue_seconds),
                "--stats-interval-seconds",
                str(self._config.stats_interval_seconds),
            ]

        raise ValueError(f"unsupported ASR runtime: {self._config.asr_runtime}")

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

    def stderr_lines(self) -> Iterator[str]:
        if self._proc is None or self._proc.stderr is None:
            raise RuntimeError("ASR process is not running")
        for line in self._proc.stderr:
            text = line.rstrip()
            if text:
                yield text

    def return_code(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.poll()

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
                self._proc.wait(timeout=2)
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
        self._stderr_reader: threading.Thread | None = None
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._opened = False
        self._listening = False
        self._closing = False
        self._inserted_partial_text = ""
        self._streaming_inserted_any = False
        self._exit_code = 0

    @property
    def listening(self) -> bool:
        with self._lock:
            return self._listening

    def open(self) -> None:
        if self._opened:
            return
        self._done.clear()
        self._exit_code = 0
        with self._lock:
            self._closing = False
        transcript_opened = False
        keyboard_opened = False
        try:
            self._transcript.open()
            transcript_opened = True
            self._keyboard.open()
            keyboard_opened = True
            if isinstance(self._keyboard, DryRunKeyboardBackend):
                self._keyboard.events.clear()
            self._asr.start()
            self._reader = threading.Thread(target=self._read_events, name="wordpipe-events", daemon=True)
            self._reader.start()
            self._stderr_reader = threading.Thread(
                target=self._read_stderr,
                name="wordpipe-stderr",
                daemon=True,
            )
            self._stderr_reader.start()
            self._opened = True
        except Exception:
            self._cleanup_failed_open(
                keyboard_opened=keyboard_opened,
                transcript_opened=transcript_opened,
            )
            raise

    def start_dictation(self) -> None:
        self.open()
        with self._lock:
            if self._listening:
                return
            self._listening = True
            self._inserted_partial_text = ""
            self._streaming_inserted_any = False
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
            with self._lock:
                self._closing = True
            self._asr.close()
            self._join_reader_threads()
        finally:
            self._keyboard.close()
            self._transcript.close()
            with self._lock:
                self._listening = False
                self._opened = False
            self._done.set()
            self._reader = None
            self._stderr_reader = None

    def _cleanup_failed_open(self, *, keyboard_opened: bool, transcript_opened: bool) -> None:
        try:
            with self._lock:
                self._closing = True
            self._asr.close()
            self._join_reader_threads()
        finally:
            if keyboard_opened:
                self._keyboard.close()
            if transcript_opened:
                self._transcript.close()
            with self._lock:
                self._listening = False
                self._opened = False
            self._reader = None
            self._stderr_reader = None

    def _join_reader_threads(self) -> None:
        current = threading.current_thread()
        for reader in (self._reader, self._stderr_reader):
            if reader is not None and reader is not current and reader.is_alive():
                reader.join(timeout=1)

    def _read_events(self) -> None:
        try:
            for item in self._asr.events():
                self._handle_event(item)
            self._handle_worker_stdout_closed()
        except Exception as exc:  # noqa: BLE001 - daemon must surface subprocess failures.
            if self._is_closing():
                self._done.set()
                return
            self._transcript.error(f"{type(exc).__name__}: {exc}")
            self._exit_code = 1
            self._done.set()

    def _read_stderr(self) -> None:
        try:
            for line in self._asr.stderr_lines():
                self._transcript.error(f"ASR worker stderr: {line}")
        except Exception as exc:  # noqa: BLE001 - stderr drain must not kill dictation.
            if not self._done.is_set():
                self._transcript.error(f"ASR stderr reader failed: {type(exc).__name__}: {exc}")

    def _handle_worker_stdout_closed(self) -> None:
        if self._done.is_set():
            return
        if self._is_closing():
            self._done.set()
            return
        return_code = self._asr_return_code()
        if return_code == 0:
            self._transcript.status("ASR worker exited")
            self._exit_code = 0
        elif return_code is None:
            self._transcript.error("ASR worker stdout closed unexpectedly")
            self._exit_code = 1
        else:
            self._transcript.error(f"ASR worker exited with status {return_code}")
            self._exit_code = return_code
        self._done.set()

    def _asr_return_code(self) -> int | None:
        return_code = getattr(self._asr, "return_code", None)
        if return_code is None:
            return None
        return return_code()

    def _is_closing(self) -> bool:
        with self._lock:
            return self._closing

    def _handle_event(self, item: dict[str, object]) -> None:
        kind = item.get("event")
        if kind == "ready":
            self._transcript.status("ASR worker ready")
        elif kind == "loading_model":
            self._transcript.status("loading ASR model")
        elif kind == "model_loaded":
            data = item.get("data")
            load_seconds = data.get("load_seconds") if isinstance(data, dict) else None
            if load_seconds is None:
                self._transcript.status("ASR model loaded")
            else:
                self._transcript.status(f"ASR model loaded in {load_seconds}s")
        elif kind == "listening":
            self._transcript.status("listening")
        elif kind == "partial":
            raw_text = str(item.get("text", ""))
            self._transcript.partial(raw_text)
            if self._config.insert_partial_text:
                self._insert_streaming_text(_normalize_text(raw_text, self._config.spoken_punctuation))
        elif kind == "commit":
            raw_text = str(item.get("text", ""))
            text = format_committed_text(_normalize_text(raw_text, self._config.spoken_punctuation))
            if text:
                self._transcript.commit(text)
                if self._config.log_metrics:
                    self._transcript.status(_format_metrics(item.get("data")))
                if self._config.insert_partial_text:
                    with self._lock:
                        streaming_inserted_any = self._streaming_inserted_any
                    if streaming_inserted_any:
                        return
                    self._transcript.status("no streamed text inserted; inserting final commit")
                    self._insert_text(text)
                else:
                    self._insert_text(text)
        elif kind == "stats":
            if self._config.log_metrics:
                self._transcript.status(_format_metrics(item.get("data")))
        elif kind == "error":
            self._transcript.error(str(item.get("message", "")))
            self._exit_code = 1
            self._done.set()
        elif kind == "stopped":
            with self._lock:
                self._listening = False
            self._transcript.status("idle")

    def _insert_streaming_text(self, text: str) -> None:
        current = text.strip(" \t")
        if text.endswith(" ") and current:
            current = f"{current} "
        if not current:
            return
        with self._lock:
            previous = self._inserted_partial_text
            if not current.startswith(previous):
                self._transcript.status("partial changed before already-inserted text; waiting for append")
                return
            suffix = current[len(previous) :]
        if suffix:
            self._insert_text(suffix)
            with self._lock:
                self._streaming_inserted_any = True
        with self._lock:
            self._inserted_partial_text = current

    def _insert_text(self, text: str) -> None:
        self._keyboard.insert_text(text)
        if isinstance(self._keyboard, DryRunKeyboardBackend):
            for rendered in self._keyboard.events:
                print(f"key: {rendered}", file=sys.stderr)
            self._keyboard.events.clear()


def _normalize_text(text: str, spoken_punctuation: bool) -> str:
    if spoken_punctuation:
        return normalize_spoken_punctuation(text)
    return text


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


def default_voice_keyboard_pid_file() -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return Path(runtime_dir) / "wordpipe" / "voice-keyboard.pid"
    return Path("/tmp") / f"wordpipe-{os.getuid()}" / "voice-keyboard.pid"


def run_signal_hotkey_daemon(
    config: DaemonConfig,
    transcript: TranscriptSink | None = None,
    pid_file: Path | None = None,
) -> int:
    keyboard: KeyboardBackend
    keyboard = DryRunKeyboardBackend() if config.dry_run_insertion else PortalKeyboardBackend()
    transcript = transcript if transcript is not None else StderrTranscriptSink()
    controller = DictationController(config, keyboard, transcript)
    done = threading.Event()
    target_pid_file = pid_file or default_voice_keyboard_pid_file()

    def toggle(_signum, _frame) -> None:  # type: ignore[no-untyped-def]
        if controller.listening:
            controller.stop_dictation()
        else:
            controller.start_dictation()

    def shutdown(_signum, _frame) -> None:  # type: ignore[no-untyped-def]
        done.set()

    previous_usr1 = signal.getsignal(signal.SIGUSR1)
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGUSR1, toggle)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        controller.open()
        target_pid_file.parent.mkdir(parents=True, exist_ok=True)
        target_pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
        transcript.status(f"voice keyboard ready pid={os.getpid()}")
        done.wait()
        return 0
    finally:
        signal.signal(signal.SIGUSR1, previous_usr1)
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        try:
            if target_pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
                target_pid_file.unlink()
        except FileNotFoundError:
            pass
        controller.close()
