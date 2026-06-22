from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import math
import queue
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Callable, TextIO

from .protocol import Event, event, parse_command


Emit = Callable[[Event], None]


@dataclass(frozen=True)
class AsrWorkerConfig:
    model_dir: Path
    provider: str = "cpu"
    num_threads: int = 2
    sample_rate: int = 16000
    feature_dim: int = 80
    decoding_method: str = "greedy_search"
    partial_interval_seconds: float = 0.10
    audio_chunk_seconds: float = 0.03
    queue_seconds: float = 2.0
    stats_interval_seconds: float = 1.0
    endpoint_rule1_min_trailing_silence: float = 0.55
    endpoint_rule2_min_trailing_silence: float = 0.35
    endpoint_rule3_min_utterance_length: float = 20.0


@dataclass(frozen=True)
class ModelLayout:
    kind: str
    model_dir: Path
    tokens: Path
    model: Path | None = None
    encoder: Path | None = None
    decoder: Path | None = None
    joiner: Path | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "model_dir": str(self.model_dir),
            "tokens": str(self.tokens),
            "model": str(self.model) if self.model else None,
            "encoder": str(self.encoder) if self.encoder else None,
            "decoder": str(self.decoder) if self.decoder else None,
            "joiner": str(self.joiner) if self.joiner else None,
        }


class JsonLineEmitter:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def __call__(self, item: Event) -> None:
        with self._lock:
            print(item.to_json(), file=self._stream, flush=True)


class AsrWorker:
    def __init__(self, config: AsrWorkerConfig, emit: Emit) -> None:
        self._config = config
        self._emit = emit
        self._session: SherpaStreamingSession | None = None

    def start(self) -> None:
        if self._session is not None:
            return
        session = SherpaStreamingSession(self._config, self._emit)
        session.start()
        self._session = session

    def stop(self) -> None:
        if self._session is None:
            self._emit(event("stopped"))
            return
        session = self._session
        self._session = None
        session.stop()
        self._emit(event("stopped"))

    def shutdown(self) -> None:
        self.stop()


class SherpaStreamingSession:
    def __init__(self, config: AsrWorkerConfig, emit: Emit) -> None:
        self._config = config
        self._emit = emit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        max_chunks = max(4, int(config.queue_seconds * 1000 / 50))
        self._audio: queue.Queue[object] = queue.Queue(maxsize=max_chunks)
        self._dropped_chunks = 0
        self._last_partial = ""
        self._last_partial_emit = 0.0
        self._last_stats_emit = 0.0
        self._last_rms = 0.0
        self._peak_rms = 0.0
        self._accepted_samples = 0
        self._decode_seconds = 0.0
        self._decode_calls = 0
        self._session_started = time.monotonic()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="wordpipe-asr", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            import sounddevice as sd

            recognizer = _create_recognizer(self._config)
            stream = recognizer.create_stream()

            def callback(indata, _frames, _time_info, status) -> None:  # type: ignore[no-untyped-def]
                if status:
                    self._emit(event("error", message=str(status)))
                try:
                    self._audio.put_nowait(indata[:, 0].copy())
                except queue.Full:
                    self._dropped_chunks += 1

            blocksize = max(1, int(self._config.sample_rate * self._config.audio_chunk_seconds))
            with sd.InputStream(
                channels=1,
                samplerate=self._config.sample_rate,
                dtype="float32",
                blocksize=blocksize,
                callback=callback,
            ):
                self._emit(event("listening"))
                while not self._stop.is_set():
                    try:
                        samples = self._audio.get(timeout=0.1)
                    except queue.Empty:
                        self._decode_ready(recognizer, stream)
                        continue

                    stream.accept_waveform(self._config.sample_rate, samples)
                    self._accepted_samples += len(samples)
                    self._record_audio_level(samples)
                    self._decode_ready(recognizer, stream)
                    self._emit_partial_if_changed(recognizer, stream)
                    self._emit_stats_if_due()
                    if recognizer.is_endpoint(stream):
                        self._commit_current(recognizer, stream)
                        recognizer.reset(stream)

            self._commit_current(recognizer, stream)
        except Exception as exc:  # noqa: BLE001 - worker must report runtime failures.
            self._emit(event("error", message=f"{type(exc).__name__}: {exc}"))

    def _decode_ready(self, recognizer: object, stream: object) -> None:
        while recognizer.is_ready(stream):  # type: ignore[attr-defined]
            started = time.monotonic()
            recognizer.decode_stream(stream)  # type: ignore[attr-defined]
            self._decode_seconds += time.monotonic() - started
            self._decode_calls += 1

    def _emit_partial_if_changed(self, recognizer: object, stream: object) -> None:
        text = _result_text(recognizer.get_result(stream))  # type: ignore[attr-defined]
        now = time.monotonic()
        if text == self._last_partial:
            return
        if now - self._last_partial_emit < self._config.partial_interval_seconds:
            return
        self._last_partial = text
        self._last_partial_emit = now
        self._emit(event("partial", text=text, data=self._metrics()))

    def _emit_stats_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_stats_emit < self._config.stats_interval_seconds:
            return
        self._last_stats_emit = now
        self._emit(event("stats", data=self._metrics()))

    def _record_audio_level(self, samples: object) -> None:
        try:
            square_sum = float((samples * samples).mean())  # type: ignore[operator, union-attr]
        except Exception:
            return
        rms = math.sqrt(max(0.0, square_sum))
        self._last_rms = rms
        self._peak_rms = max(self._peak_rms, rms)

    def _commit_current(self, recognizer: object, stream: object) -> None:
        text = _result_text(recognizer.get_result(stream)).strip()  # type: ignore[attr-defined]
        if not text:
            return
        self._emit(
            event(
                "commit",
                text=text,
                data=self._metrics(),
            )
        )
        self._last_partial = ""
        self._last_partial_emit = 0.0
        self._accepted_samples = 0
        self._decode_seconds = 0.0
        self._decode_calls = 0
        self._last_rms = 0.0
        self._peak_rms = 0.0
        self._session_started = time.monotonic()

    def _metrics(self) -> dict[str, float | int]:
        audio_seconds = self._accepted_samples / self._config.sample_rate
        elapsed_seconds = time.monotonic() - self._session_started
        return {
            "audio_seconds": round(audio_seconds, 3),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "decode_seconds": round(self._decode_seconds, 3),
            "decode_calls": self._decode_calls,
            "dropped_audio_chunks": self._dropped_chunks,
            "last_rms": round(self._last_rms, 5),
            "peak_rms": round(self._peak_rms, 5),
            "real_time_factor": round(self._decode_seconds / audio_seconds, 3)
            if audio_seconds > 0
            else 0.0,
        }


def _result_text(result: object) -> str:
    text = getattr(result, "text", result)
    return str(text or "").strip()


def _create_recognizer(config: AsrWorkerConfig) -> object:
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "sherpa_onnx is not installed; install the project with the asr extra"
        ) from exc

    layout = discover_model_layout(config.model_dir)

    common = {
        "tokens": str(layout.tokens),
        "num_threads": config.num_threads,
        "sample_rate": config.sample_rate,
        "feature_dim": config.feature_dim,
        "decoding_method": config.decoding_method,
        "provider": config.provider,
        "enable_endpoint_detection": True,
        "rule1_min_trailing_silence": config.endpoint_rule1_min_trailing_silence,
        "rule2_min_trailing_silence": config.endpoint_rule2_min_trailing_silence,
        "rule3_min_utterance_length": config.endpoint_rule3_min_utterance_length,
    }

    recognizer_cls = sherpa_onnx.OnlineRecognizer
    if layout.kind == "transducer":
        return _call_factory(
            recognizer_cls.from_transducer,
            {
                **common,
                "encoder": str(layout.encoder),
                "decoder": str(layout.decoder),
                "joiner": str(layout.joiner),
            },
        )

    if layout.kind == "nemo_ctc" and hasattr(recognizer_cls, "from_nemo_ctc"):
        return _call_factory(
            recognizer_cls.from_nemo_ctc,
            {
                **common,
                "model": str(layout.model),
            },
        )

    raise RuntimeError(f"sherpa-onnx does not support discovered layout: {layout.kind}")


def discover_model_layout(model_dir: Path) -> ModelLayout:
    resolved = model_dir.expanduser().resolve()
    tokens = resolved / "tokens.txt"
    if not tokens.exists():
        raise FileNotFoundError(f"missing tokens file: {tokens}")

    encoder = _find_one(resolved, "encoder*.onnx")
    decoder = _find_one(resolved, "decoder*.onnx")
    joiner = _find_one(resolved, "joiner*.onnx")
    if encoder and decoder and joiner:
        return ModelLayout(
            kind="transducer",
            model_dir=resolved,
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
        )

    onnx_files = sorted(resolved.glob("*.onnx"))
    if len(onnx_files) == 1:
        return ModelLayout(
            kind="nemo_ctc",
            model_dir=resolved,
            tokens=tokens,
            model=onnx_files[0],
        )

    raise FileNotFoundError(
        "could not identify a supported sherpa-onnx streaming model layout in "
        f"{resolved}"
    )


def render_model_info(model_dir: Path) -> str:
    return json.dumps(discover_model_layout(model_dir).to_dict(), indent=2, sort_keys=True)


def transcribe_wav_file(config: AsrWorkerConfig, wav_path: Path) -> tuple[str, dict[str, float | int]]:
    samples, sample_rate = read_wav_mono_float32(wav_path)
    if sample_rate != config.sample_rate:
        raise ValueError(f"expected {config.sample_rate} Hz audio, got {sample_rate} Hz")

    started = time.monotonic()
    recognizer = _create_recognizer(config)
    stream = recognizer.create_stream()
    decode_seconds = 0.0
    decode_calls = 0
    chunk_size = max(1, int(config.sample_rate * 0.1))
    for offset in range(0, len(samples), chunk_size):
        stream.accept_waveform(config.sample_rate, samples[offset : offset + chunk_size])
        while recognizer.is_ready(stream):
            decode_started = time.monotonic()
            recognizer.decode_stream(stream)
            decode_seconds += time.monotonic() - decode_started
            decode_calls += 1

    stream.input_finished()
    while recognizer.is_ready(stream):
        decode_started = time.monotonic()
        recognizer.decode_stream(stream)
        decode_seconds += time.monotonic() - decode_started
        decode_calls += 1
    audio_seconds = len(samples) / config.sample_rate
    metrics = {
        "audio_seconds": round(audio_seconds, 3),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "decode_seconds": round(decode_seconds, 3),
        "decode_calls": decode_calls,
        "real_time_factor": round(decode_seconds / audio_seconds, 3)
        if audio_seconds > 0
        else 0.0,
    }
    return _result_text(recognizer.get_result(stream)), metrics


def read_wav_mono_float32(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if channels != 1:
        raise ValueError(f"expected mono WAV, got {channels} channels")
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got sample width {sample_width}")

    samples = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    return samples, sample_rate


def _find_one(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _call_factory(factory: Callable[..., object], kwargs: dict[str, object]) -> object:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return factory(**kwargs)

    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return factory(**accepted)


def run_stdio_worker(config: AsrWorkerConfig) -> int:
    emitter = JsonLineEmitter(sys.stdout)
    worker = AsrWorker(config, emitter)
    emitter(event("ready"))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            command = parse_command(line)
        except ValueError as exc:
            emitter(event("error", message=str(exc)))
            continue

        if command.command == "start":
            worker.start()
        elif command.command == "stop":
            worker.stop()
        elif command.command == "shutdown":
            worker.shutdown()
            return 0

    worker.shutdown()
    return 0
