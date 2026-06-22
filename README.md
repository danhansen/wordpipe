# Wordpipe

Wordpipe is a Wayland-only GNOME dictation app built around true streaming
speech recognition. The ASR runtime is pivoting to a Rust worker based on
`parakeet-rs`; the earlier sherpa-onnx worker remains available for legacy
diagnostics.

## Direction

- GNOME-first Linux desktop integration.
- Wayland only; no X11 tooling.
- Streaming ASR with `parakeet-rs`.
- Target model family: Parakeet/Nemotron cache-aware streaming ASR.
- No external VAD for the MVP.
- Endpoint detection is disabled by default while raw continuous streaming is
  evaluated.
- Partial recognition appears in Wordpipe UI; committed text is inserted when
  dictation stops.

See [docs/architecture.md](docs/architecture.md) for the current design plan.

## Current MVP

The current implementation provides:

- `wordpipe probe` capability checks for GNOME, portals, and Python modules.
- `wordpipe-parakeet-worker` Rust newline-JSON streaming worker.
- `wordpipe asr-worker` legacy sherpa-onnx newline-JSON worker.
- `wordpipe type-text` keyboard insertion through the RemoteDesktop portal.
- `wordpipe daemon` MVP loop that connects the ASR worker to text insertion.
- `wordpipe hotkey-daemon` manual or GlobalShortcuts-controlled dictation.
- Optional libadwaita/GTK live transcript overlay.

The GNOME Shell extension and top-bar indicator are not built yet.

## Local Development

This workspace contains a mounted placeholder `.git` directory, so the real Git
metadata lives in `.wordpipe.git`. Use:

```sh
git --git-dir=.wordpipe.git --work-tree=. status
```

Run tests:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests
```

Build the Rust Parakeet worker:

```sh
cargo build --release -p wordpipe-parakeet-worker
```

After creating `.venv`, the `scripts/wordpipe-dev` wrapper runs the local source
tree without repeating `PYTHONPATH=src .venv/bin/python -m wordpipe`:

```sh
scripts/wordpipe-dev probe
```

Run the capability probe:

```sh
PYTHONPATH=src python3 -m wordpipe probe
```

Inspect a downloaded sherpa-onnx model directory:

```sh
PYTHONPATH=src python3 -m wordpipe model-info --model-dir /path/to/model
```

Run offline decoding against a WAV file:

```sh
PYTHONPATH=src python3 -m wordpipe transcribe-file \
  --model-dir models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 \
  --wav models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/test_wavs/en.wav
```

Run live partial-only testing with RTF metrics using the Rust Parakeet runtime:

```sh
cargo build --release -p wordpipe-parakeet-worker
scripts/wordpipe-dev listen-test \
  --model-dir /path/to/parakeet-nemotron-streaming-model
```

This opens the microphone and prints `partial` and `commit` events without
inserting text into any app. It also prints periodic `stats` lines with RTF,
audio level, and dropped-chunk counts. When the recognizer has a current
hypothesis, each stats tick also repeats it as a `partial` line, so you can see
stable partial text even when it has not changed. Use Ctrl+C to stop.
The default Parakeet runtime takes raw continuous mic audio into ASR and commits
the accumulated transcript when dictation stops. The legacy sherpa runtime can
still be selected with `--asr-runtime sherpa`.

List input devices:

```sh
scripts/wordpipe-dev audio-devices
```

Try a specific input device:

```sh
scripts/wordpipe-dev listen-test \
  --input-device 12 \
  --model-dir /path/to/parakeet-nemotron-streaming-model
```

Record what Wordpipe is hearing:

```sh
scripts/wordpipe-dev record-test --duration 5 --output /tmp/wordpipe-spoken.wav
scripts/wordpipe-dev transcribe-file \
  --model-dir models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 \
  --wav /tmp/wordpipe-spoken.wav
```

If the recorded WAV transcribes but `listen-test` does not produce partials, the
problem is streaming throughput. If the WAV does not transcribe, the issue is
audio capture, device selection, level, or model suitability for the speech.

Test streaming behavior from a known-good WAV:

```sh
scripts/wordpipe-dev stream-file-test \
  --model-dir models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 \
  --wav /tmp/wordpipe-spoken.wav
```

This should print partials if the model emits them in streaming mode.

Dry-run text insertion:

```sh
PYTHONPATH=src python3 -m wordpipe type-text --dry-run "hello world"
```

Create a config file:

```sh
mkdir -p ~/.config/wordpipe
PYTHONPATH=src python3 -m wordpipe config-example > ~/.config/wordpipe/config.toml
```

Run the MVP daemon:

```sh
PYTHONPATH=src python3 -m wordpipe daemon \
  --model-dir /path/to/parakeet-nemotron-streaming-model
```

Use `--dry-run-insertion` to exercise ASR without opening a portal keyboard
session.

When `~/.config/wordpipe/config.toml` contains `model_dir`, `daemon` and
`hotkey-daemon` can run without `--model-dir`. CLI flags override config values.

Packaging templates live under `packaging/`:

- `packaging/applications/dev.wordpipe.Wordpipe.desktop`
- `packaging/systemd/wordpipe.service`

They assume `wordpipe` is installed on `PATH` and configuration exists at
`~/.config/wordpipe/config.toml`.

Run the hotkey-controlled daemon:

```sh
PYTHONPATH=src python3 -m wordpipe hotkey-daemon \
  --model-dir /path/to/parakeet-nemotron-streaming-model \
  --mode hold \
  --shortcut 'CTRL+ALT+space' \
  --overlay gtk
```

For development without the GlobalShortcuts portal:

```sh
PYTHONPATH=src python3 -m wordpipe hotkey-daemon \
  --model-dir /path/to/parakeet-nemotron-streaming-model \
  --manual-hotkey \
  --dry-run-insertion
```

Manual commands are `down`, `up`, `toggle`, and `quit`.

## Runtime Dependencies

Install Python ASR dependencies only when using the legacy sherpa worker:

```sh
python3 -m pip install '.[asr]'
```

The local development environment has been smoke-tested with
`sherpa-onnx==1.13.3` on Python 3.14 for the legacy worker.

The default Rust runtime uses `parakeet-rs`. Build it with:

```sh
cargo build --release -p wordpipe-parakeet-worker
```

`listen-test`, `daemon`, and `hotkey-daemon` look for
`target/release/wordpipe-parakeet-worker` first, then the debug binary, then
`wordpipe-parakeet-worker` on `PATH`. Use `--asr-worker-path` to point at a
custom binary.

Download the legacy sherpa 560 ms int8 Nemotron model:

```sh
PYTHONPATH=src python3 -m wordpipe download-model
```

This writes to `models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/`
by default. The repository is:

```text
csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11
```

The legacy sherpa model directory must contain `tokens.txt` and either a single
`.onnx` model for the Nemotron CTC path or `encoder*.onnx`, `decoder*.onnx`,
and `joiner*.onnx` for a transducer layout.

The default Parakeet/Nemotron runtime expects the model layout used by
`parakeet-rs`: `encoder.onnx`, any associated external data file,
`decoder_joint.onnx`, and `tokenizer.model`.

The tested int8 English model is:

```text
models/nemotron-speech-streaming-en-0.6b-int8/
```

On the current Ivy Bridge CPU, the Rust worker uses dynamic ONNX Runtime
loading so it can reuse the non-AVX2 `libonnxruntime.so` bundled with
`sherpa_onnx`. `scripts/wordpipe-dev`, `listen-test`, and daemon launch paths
set `ORT_DYLIB_PATH` automatically when that library is present in `.venv`.

## Live Validation

Validated in GNOME 50.2 on Wayland:

- RemoteDesktop portal text insertion into a focused app.
- Manual hotkey end-to-end dictation with the Adwaita/GTK overlay.
- Live microphone capture reaching the ASR `listening` state.
- Offline decoding with the downloaded Nemotron int8 model.

## Performance Notes

The default runtime is:

```text
asr_runtime = "parakeet"
num_threads = 2
queue_seconds = 10.0
```

The Rust worker feeds Nemotron in 560 ms chunks, matching the model's streaming
stride. It reports RTF, audio level, and dropped audio chunks through the same
JSON events as the legacy worker.

On the current test machine, both tested CPU runtimes decode slower than
realtime. The Parakeet int8 English model streams correctly from the known
test WAV and emits partials/commit events, but the release worker measures
about 2.5 RTF per 560 ms chunk and about 3.1 RTF including final flush.

The GTK overlay prefers libadwaita (`Adw 1`) and falls back to plain GTK 4 if
libadwaita is not available. Non-UI daemon paths do not require GTK.

Committed text converts common spoken punctuation commands by default:

```text
hello comma world period -> hello, world.
new line -> Enter
new paragraph -> blank line
```

Use `--no-spoken-punctuation` to insert raw ASR output.
