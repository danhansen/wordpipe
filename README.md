# Wordpipe

Wordpipe is a Wayland-only GNOME dictation app built around true streaming
speech recognition with sherpa-onnx.

## Direction

- GNOME-first Linux desktop integration.
- Wayland only; no X11 tooling.
- Streaming ASR with `sherpa-onnx`.
- Target model: `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8`.
- No external VAD for the MVP.
- Endpoint detection creates phrase commit boundaries while dictation continues.
- Partial recognition appears in Wordpipe UI; committed phrases are inserted into
  the focused app.

See [docs/architecture.md](docs/architecture.md) for the current design plan.

## Current MVP

The current implementation provides:

- `wordpipe probe` capability checks for GNOME, portals, and Python modules.
- `wordpipe asr-worker` newline-JSON streaming worker protocol.
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

Run live partial-only testing with RTF metrics:

```sh
scripts/wordpipe-dev listen-test \
  --model-dir models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11
```

This opens the microphone and prints `partial` and `commit` events without
inserting text into any app. It also prints periodic `stats` lines with RTF,
audio level, and dropped-chunk counts. When the recognizer has a current
hypothesis, each stats tick also repeats it as a `partial` line, so you can see
stable partial text even when it has not changed. Use Ctrl+C to stop.

List input devices:

```sh
scripts/wordpipe-dev audio-devices
```

Try a specific input device:

```sh
scripts/wordpipe-dev listen-test \
  --input-device 12 \
  --model-dir models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11
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
  --model-dir /path/to/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8
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
  --model-dir /path/to/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8 \
  --mode hold \
  --shortcut 'CTRL+ALT+space' \
  --overlay gtk
```

For development without the GlobalShortcuts portal:

```sh
PYTHONPATH=src python3 -m wordpipe hotkey-daemon \
  --model-dir /path/to/model \
  --manual-hotkey \
  --dry-run-insertion
```

Manual commands are `down`, `up`, `toggle`, and `quit`.

## Runtime Dependencies

Install ASR dependencies in the environment that runs Wordpipe:

```sh
python3 -m pip install '.[asr]'
```

The local development environment has been smoke-tested with
`sherpa-onnx==1.13.3` on Python 3.14.

Download the default 560 ms int8 Nemotron model:

```sh
PYTHONPATH=src python3 -m wordpipe download-model
```

This writes to `models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/`
by default. The repository is:

```text
csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11
```

The model directory must contain `tokens.txt` and either a single `.onnx` model
for the Nemotron CTC path or `encoder*.onnx`, `decoder*.onnx`, and
`joiner*.onnx` for a transducer layout.

## Live Validation

Validated in GNOME 50.2 on Wayland:

- RemoteDesktop portal text insertion into a focused app.
- Manual hotkey end-to-end dictation with the Adwaita/GTK overlay.
- Live microphone capture reaching the ASR `listening` state.
- Offline decoding with the downloaded Nemotron int8 model.

## Performance Notes

The default CPU tuning is currently:

```text
num_threads = 2
audio_chunk_seconds = 0.03
queue_seconds = 10.0
partial_interval_seconds = 0.10
endpoint_rule1_min_trailing_silence = 0.55
endpoint_rule2_min_trailing_silence = 0.35
```

The live ASR queue is intentionally larger than the audio chunk size requires,
because this model can run slower than realtime on the test CPU. Buffering avoids
dropping speech context; it trades some latency for recognition continuity.

On the current test machine, the 560 ms int8 model decodes slower than realtime
on CPU. The best measured CPU thread count was 2 threads; higher counts were
slower. GPU acceleration remains exposed through `--provider`, but this machine
does not have a CUDA-capable GPU and the installed wheel did not expose a GPU
provider.

The GTK overlay prefers libadwaita (`Adw 1`) and falls back to plain GTK 4 if
libadwaita is not available. Non-UI daemon paths do not require GTK.

Committed text converts common spoken punctuation commands by default:

```text
hello comma world period -> hello, world.
new line -> Enter
new paragraph -> blank line
```

Use `--no-spoken-punctuation` to insert raw ASR output.
