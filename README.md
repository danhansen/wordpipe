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
The Rust worker defaults to ONNX Runtime's `all` graph optimization level; use
`--graph-optimization` only for ablations or debugging.

Optimization work is tracked in
[docs/optimization-experiments.md](docs/optimization-experiments.md). The
Sayboard optimization inventory and harvest results are in
[docs/sayboard-optimization-harvest.md](docs/sayboard-optimization-harvest.md).

Inspect ONNX graphs and ORT optimization effects:

```sh
.venv/bin/python scripts/ort_graph_diagnostics.py \
  models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56/encoder.onnx \
  --json-out build/ort-diagnostics/encoder-summary.json
```

For smaller graphs, or when you are comfortable spending the memory to let ORT
load and serialize an optimized model, add `--emit-optimized --opt-level all`.
The resulting summary makes ORT fusions visible, such as
`DynamicQuantizeLinear + MatMulInteger` becoming `DynamicQuantizeMatMul`.

If you run `target/release/wordpipe-parakeet-worker` directly, set
`ORT_DYLIB_PATH` to the ONNX Runtime library from the local Python wheel. The
`ort` crate's default runtime can hang while loading this encoder on the current
machine:

```sh
ORT_DYLIB_PATH="$PWD/.venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so.1.27.0" \
  target/release/wordpipe-parakeet-worker \
  --model-dir /path/to/parakeet-nemotron-streaming-model \
  --wav /path/to/test.wav
```

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
loading. `scripts/wordpipe-dev`, `listen-test`, and daemon launch paths set
`ORT_DYLIB_PATH` automatically when a local `onnxruntime` or `sherpa_onnx`
library is present in `.venv`; the `onnxruntime` wheel library is preferred
because it loads the projected-cache Nemotron encoder reliably here.

### Building A Wordpipe Nemotron Model

The current best export path is codified as a thin wrapper around the individual
phase scripts:

```sh
.venv/bin/python scripts/build_nemotron_wordpipe_model.py \
  /path/to/model.nemo \
  models/nemotron-wordpipe-ffn-fp32 \
  --work-dir build/nemotron-wordpipe-pipeline
```

Use `--force` to overwrite an existing work/output directory. Use `--dry-run`
to print the phase commands without running them.

The wrapper deliberately keeps the phases separate:

- `export_nemotron_parakeet_optimized.py --export-only` exports FP32 ONNX from
  NeMo and then exits before quantization so Torch/NeMo memory is released.
- `transform_nemotron_parakeet_export.py` applies dynamic QUInt8 quantization
  and rewrites the encoder to use projected K/V cache.
- `build_nemotron_fixed_shape_model.py` specializes the streaming graph to the
  current c56 runtime shape and serializes ORT's optimized encoder graph.
- `dequantize_nemotron_matmul_blocks.py --include /feed_forward` rewrites FFN
  MatMul/Gemm blocks back to FP32, the fastest validated variant on the current
  Ivy Bridge test machine.
- `--fp32-decoder` keeps `decoder_joint.onnx` as the FP32 NeMo export while the
  encoder follows the quantized/projected-cache pipeline. It is experimental:
  the local LibriSpeech sample showed a small throughput win with one additional
  strict spelling WER error.

Important defaults:

```text
left_context = 56
right_context = 6
input_frames = 65
output_frames = 7
cache_len = 56
hidden_dim = 1024
ort_optimize_final = extended
```

The final output directory contains the runtime model files:

```text
encoder.onnx
decoder_joint.onnx
tokenizer.model
config.json
```

The wrapper supports `--start-at` and `--stop-after` for resuming or debugging
individual phases. For example, after a successful FP32 export:

```sh
.venv/bin/python scripts/build_nemotron_wordpipe_model.py \
  /path/to/model.nemo \
  models/nemotron-wordpipe-ffn-fp32 \
  --work-dir build/nemotron-wordpipe-pipeline \
  --start-at transform
```

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

The workers feed Nemotron in 560 ms chunks, matching the model's streaming
stride. File tests feed three synthetic silence chunks by default so streaming
models can emit trailing tokens before the final commit.

Metrics report both `audio_seconds` for real input and `processed_audio_seconds`
for real input plus padding/flush audio. `real_time_factor` is calculated from
processed audio so synthetic flush work is accounted for fairly;
`real_audio_real_time_factor` keeps the stricter real-input denominator visible.

On the current test machine, the c56 Parakeet int8 export with ORT graph
optimization `all` decodes the known sherpa English test WAV at about 0.94 RTF
over processed audio with the final flush included. The legacy sherpa int8 path
is about 0.99 RTF on the same test and still misses the trailing "gold" token.

The GTK overlay prefers libadwaita (`Adw 1`) and falls back to plain GTK 4 if
libadwaita is not available. Non-UI daemon paths do not require GTK.

Committed text converts common spoken punctuation commands by default:

```text
hello comma world period -> hello, world.
new line -> Enter
new paragraph -> blank line
```

Use `--no-spoken-punctuation` to insert raw ASR output.
