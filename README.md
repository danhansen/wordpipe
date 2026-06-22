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

The GNOME Shell extension, top-bar indicator, and live overlay are not built yet.

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

Run the capability probe:

```sh
PYTHONPATH=src python3 -m wordpipe probe
```

Inspect a downloaded sherpa-onnx model directory:

```sh
PYTHONPATH=src python3 -m wordpipe model-info --model-dir /path/to/model
```

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

The model directory must contain `tokens.txt` and either a single `.onnx` model
for the Nemotron CTC path or `encoder*.onnx`, `decoder*.onnx`, and
`joiner*.onnx` for a transducer layout.

The GTK overlay prefers libadwaita (`Adw 1`) and falls back to plain GTK 4 if
libadwaita is not available. Non-UI daemon paths do not require GTK.

Committed text converts common spoken punctuation commands by default:

```text
hello comma world period -> hello, world.
new line -> Enter
new paragraph -> blank line
```

Use `--no-spoken-punctuation` to insert raw ASR output.
