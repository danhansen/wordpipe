# Wordpipe Architecture

## Product Shape

Wordpipe is a Wayland-only GNOME dictation app. It should behave like a
consented virtual keyboard driven by streaming ASR, not like a screen scraper or
X11-style automation tool.

The first usable version prioritizes reliable raw streaming diagnostics and
committed text insertion over live cursor replacement. Partial results are
visible in Wordpipe's own UI, and current text is sent to the focused
application when dictation stops.

## Locked Decisions

- GNOME-first.
- Wayland-only.
- No `xdotool` or X11 fallback path.
- No external VAD for the MVP.
- Use a Rust `parakeet-rs` ASR worker as the default runtime.
- Keep the Python sherpa-onnx worker only as a legacy diagnostic path.
- Avoid endpointing as a commit boundary for the default runtime.
- Commit non-empty partial text when dictation is stopped.
- ASR runs out-of-process to avoid making the Python GIL a core design risk.
- Text insertion uses virtual-keyboard semantics through XDG portals/libei.
- Clipboard paste is only a debug or emergency fallback.

## Process Model

```text
GNOME integration
  - global hotkey
  - status indicator
  - optional live transcript surface

wordpipe-daemon
  - owns dictation session state
  - manages portal permissions
  - starts/stops ASR process
  - receives partial/final transcript events
  - inserts committed text through the keyboard injection backend

wordpipe-voice-keyboard
  - app-facing global-hotkey mode
  - keeps focus in the target application
  - uses the GTK overlay for visible partial/status text
  - commits recognized text into the focused text field on stop

wordpipe-app
  - GTK/libadwaita control surface
  - shows ASR load/listening/error state
  - shows live partial text, last commit, and RTF metrics
  - reuses wordpipe-daemon's controller and insertion backend

wordpipe-parakeet-worker
  - loads Parakeet/Nemotron model through parakeet-rs
  - captures microphone audio while active
  - feeds 560 ms chunks into Nemotron
  - emits partial and committed transcript events
```

The ASR worker is a subprocess speaking newline-delimited JSON. The GNOME
daemon does not depend on the model runtime; it can spawn the default Rust
Parakeet worker or the legacy Python sherpa worker.

## ASR Session Behavior

```text
dictation starts
  open microphone
  create online recognizer stream
  accept audio continuously

speech arrives
  decode fixed 560 ms Nemotron chunks
  emit partial transcript updates

dictation stops
  commit non-empty current partial
  close microphone
  stop or idle ASR worker

legacy sherpa diagnostic mode
  optionally test endpoint detection/reset behavior
```

No external VAD is used. The streaming ASR model receives the audio stream
directly.

Current low-latency defaults:

- 560 ms Nemotron chunks in the Rust worker
- 10 s audio queue before dropping microphone chunks
- endpoint detection disabled
- 2 CPU threads, based on local benchmark results

## Text Insertion

Primary target:

- XDG Desktop Portal `RemoteDesktop` keyboard session.
- Prefer `ConnectToEIS` / libei when available.
- Fall back to portal keyboard notification methods where practical.

The insertion backend should expose a text-oriented interface internally, even
though the implementation sends key events. The first version can assume US
keyboard layout for plain English text and punctuation.

Later work:

- layout-aware key generation
- richer Unicode insertion strategy
- input-method protocol investigation if virtual keyboard events become too
  limiting

## Hotkey And UI

The design supports both hold-to-dictate and toggle-to-dictate. The MVP should
choose one default and keep the state machine compatible with both.

Recommended initial UI:

- global-hotkey voice-keyboard launcher for actual focused-app text input
- GTK/libadwaita app window for profile/status/control
- top-bar status indicator for idle/listening/permission/error states
- small live transcript overlay for partial text
- committed text feedback after dictation stops

Partial text is never typed into the target app in v1.

The current `voice-keyboard` command is the primary insertion path because it
does not steal focus from the target text field. The app window and overlay
backend use libadwaita when available, with a GTK 4 fallback. A GNOME Shell
top-bar indicator remains future work.

## Configuration

The app and daemon read `~/.config/wordpipe/config.toml` by default.
Configuration holds model path/profile, ASR runtime, worker path, provider,
thread count, overlay, hotkey mode, shortcut, spoken-punctuation behavior, and
dry-run insertion. CLI flags override file values.

## Model

The default runtime expects the Parakeet/Nemotron model layout used by
`parakeet-rs`:

```text
encoder.onnx
encoder.onnx.data
decoder_joint.onnx
tokenizer.model
```

The earlier sherpa-onnx int8 model remains useful for legacy diagnostics, but it
is not the default runtime target.

Wordpipe keeps two app-level model profiles:

- `fast`: FP32 projected-cache export, best validated speed/accuracy, largest
  footprint.
- `compact`: dynamic-int8 projected-cache export with fixed shapes and
  ORT-format startup, smaller footprint and sub-second load target.

`wordpipe model-install --profile fast|compact` downloads or reuses the source
`.nemo` checkpoint and invokes the reproducible NeMo -> Wordpipe export wrapper.
Both profiles can coexist under `model_root`; changing `model_profile` or
passing `--model-profile fast|compact` selects which one the app and daemon load
when `model_dir` is not explicitly set. If the selected profile is missing, the
install path is the same whether this is first-run setup or a later attempt to
try the other model: build that profile from the cached or downloaded source
`.nemo`.

## Performance

`listen-test` is the primary live tuning mode. It opens the microphone, prints
partial results, and reports realtime factor (RTF) without inserting text.
Endpoint detection is disabled by default across live paths so raw continuous
ASR behavior is visible without phrase-boundary resets.
`audio-devices` and `record-test` are diagnostic commands for validating the
capture device independently from ASR.
`stream-file-test` feeds a known WAV through the streaming recognizer and is the
primary check for whether the model emits partial hypotheses before finalization.

The current test machine exposes only Intel HD Graphics 4000, so CPU tuning and
cache-aware runtime work are the practical performance paths. Its Ivy Bridge
CPU lacks AVX2, so the Rust worker uses dynamic ONNX Runtime loading and reuses
the non-AVX2 `libonnxruntime.so` from the local `sherpa_onnx` installation.

## Packaging

The repository includes early templates for a desktop entry and user systemd
service. These are intended for local GNOME session testing after `wordpipe` is
installed on `PATH`; they are not a complete distro package.

## MVP Milestones

1. Runtime capability probe
   - GNOME version
   - portal availability
   - GlobalShortcuts availability
   - RemoteDesktop keyboard availability
   - EIS/libei availability
   - status: implemented and live-validated on GNOME 50.2 Wayland

2. Streaming ASR spike
   - load Parakeet/Nemotron model
   - stream microphone audio
   - print raw partial text
   - log real-time factor, queue depth, and commit latency
   - status: Rust worker loads the int8 Parakeet/Nemotron model and streams a
     known WAV through partial/stats/commit events; live mic validation is
     blocked in the current tool session because PipeWire/Pulse input access is
     unavailable

3. ASR process protocol
   - newline-delimited JSON over stdio or Unix socket
   - events: `partial`, `commit`, `error`, `ready`
   - commands: `start`, `stop`, `shutdown`
   - status: implemented over stdio

4. Keyboard insertion spike
   - portal permission flow
   - insert simple ASCII phrases into focused GNOME apps
   - test Text Editor, Terminal, Firefox, LibreOffice
   - status: RemoteDesktop `NotifyKeyboardKeysym` backend implemented and
     live-validated

5. GNOME hotkey and status
   - global shortcut or shell extension trigger
   - daemon session state
   - visible listening/error state
   - status: GlobalShortcuts daemon path implemented; GTK/libadwaita app window
     provides visual status; top-bar indicator not implemented

6. First integrated dictation
   - hotkey controls dictation
   - overlay shows partials
   - stop commits non-empty partial
   - status: CLI daemon, hotkey daemon, app control window, and optional
     Adwaita/GTK overlay implemented; top-bar indicator not implemented

## Validation Matrix

- GNOME Text Editor
- Terminal
- Firefox text fields
- LibreOffice Writer
- Flatpak app text field
- password field behavior, which should not receive dictated text unless the
  user explicitly accepts that risk later

## Open Questions

- Default hotkey mode: hold-to-dictate, toggle-to-dictate, or both.
- Whether the first GNOME integration should use portals only or include a Shell
  extension immediately.
- Model installation: managed `model-install` path builds source NeMo into the
  selected `fast` or `compact` profile; graphical install controls remain a
  UI follow-up.
- Whether spoken punctuation should remain always-on by default after live
  testing.
