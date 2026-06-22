from __future__ import annotations

from dataclasses import dataclass
import inspect
import queue
import sys
import threading
import time
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
    partial_interval_seconds: float = 0.15
    queue_seconds: float = 2.0


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
        self._emit(event("listening"))

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

            blocksize = max(1, int(self._config.sample_rate * 0.05))
            with sd.InputStream(
                channels=1,
                samplerate=self._config.sample_rate,
                dtype="float32",
                blocksize=blocksize,
                callback=callback,
            ):
                while not self._stop.is_set():
                    try:
                        samples = self._audio.get(timeout=0.1)
                    except queue.Empty:
                        self._decode_ready(recognizer, stream)
                        continue

                    stream.accept_waveform(self._config.sample_rate, samples)
                    self._decode_ready(recognizer, stream)
                    self._emit_partial_if_changed(recognizer, stream)
                    if recognizer.is_endpoint(stream):
                        self._commit_current(recognizer, stream)
                        recognizer.reset(stream)

            self._commit_current(recognizer, stream)
        except Exception as exc:  # noqa: BLE001 - worker must report runtime failures.
            self._emit(event("error", message=f"{type(exc).__name__}: {exc}"))

    def _decode_ready(self, recognizer: object, stream: object) -> None:
        while recognizer.is_ready(stream):  # type: ignore[attr-defined]
            recognizer.decode_stream(stream)  # type: ignore[attr-defined]

    def _emit_partial_if_changed(self, recognizer: object, stream: object) -> None:
        text = _result_text(recognizer.get_result(stream))  # type: ignore[attr-defined]
        now = time.monotonic()
        if text == self._last_partial:
            return
        if now - self._last_partial_emit < self._config.partial_interval_seconds:
            return
        self._last_partial = text
        self._last_partial_emit = now
        self._emit(event("partial", text=text))

    def _commit_current(self, recognizer: object, stream: object) -> None:
        text = _result_text(recognizer.get_result(stream)).strip()  # type: ignore[attr-defined]
        if not text:
            return
        self._emit(
            event(
                "commit",
                text=text,
                data={"dropped_audio_chunks": self._dropped_chunks},
            )
        )
        self._last_partial = ""
        self._last_partial_emit = 0.0


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

    model_dir = config.model_dir.expanduser().resolve()
    tokens = model_dir / "tokens.txt"
    if not tokens.exists():
        raise FileNotFoundError(f"missing tokens file: {tokens}")

    onnx_files = sorted(model_dir.glob("*.onnx"))
    encoder = _find_one(model_dir, "encoder*.onnx")
    decoder = _find_one(model_dir, "decoder*.onnx")
    joiner = _find_one(model_dir, "joiner*.onnx")

    common = {
        "tokens": str(tokens),
        "num_threads": config.num_threads,
        "sample_rate": config.sample_rate,
        "feature_dim": config.feature_dim,
        "decoding_method": config.decoding_method,
        "provider": config.provider,
        "enable_endpoint_detection": True,
        "rule1_min_trailing_silence": 1.2,
        "rule2_min_trailing_silence": 0.8,
        "rule3_min_utterance_length": 20.0,
    }

    recognizer_cls = sherpa_onnx.OnlineRecognizer
    if encoder and decoder and joiner:
        return _call_factory(
            recognizer_cls.from_transducer,
            {
                **common,
                "encoder": str(encoder),
                "decoder": str(decoder),
                "joiner": str(joiner),
            },
        )

    if len(onnx_files) == 1 and hasattr(recognizer_cls, "from_nemo_ctc"):
        return _call_factory(
            recognizer_cls.from_nemo_ctc,
            {
                **common,
                "model": str(onnx_files[0]),
            },
        )

    raise FileNotFoundError(
        "could not identify a supported sherpa-onnx streaming model layout in "
        f"{model_dir}"
    )


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
