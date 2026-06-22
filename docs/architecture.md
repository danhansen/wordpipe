# Wordpipe Architecture

## Product Shape

Wordpipe is a Wayland-only GNOME dictation app. It should behave like a
consented virtual keyboard driven by streaming ASR, not like a screen scraper or
X11-style automation tool.

The first usable version prioritizes reliable committed text insertion over live
cursor replacement. Partial results are visible in Wordpipe's own UI, and only
endpoint/final text is sent to the focused application.

## Locked Decisions

- GNOME-first.
- Wayland-only.
- No `xdotool` or X11 fallback path.
- No external VAD for the MVP.
- Use sherpa-onnx streaming endpoint detection for commit boundaries.
- Endpoint detection commits the current phrase and resets the recognizer stream.
- Endpoint detection does not stop dictation.
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

wordpipe-asr
  - loads sherpa-onnx model
  - captures microphone audio while active
  - feeds audio chunks into OnlineRecognizer
  - emits partial and committed transcript events
```

The ASR worker can start as Python using the sherpa-onnx Python API. The process
boundary lets us replace it later with C++ or Rust without rewriting GNOME
integration.

## ASR Session Behavior

```text
dictation starts
  open microphone
  create online recognizer stream
  accept audio continuously

speech arrives
  decode as sherpa-onnx becomes ready
  emit partial transcript updates

endpoint fires
  emit committed phrase
  reset recognizer stream
  continue listening

dictation stops
  commit non-empty current partial
  close microphone
  stop or idle ASR worker
```

No external VAD is used. The streaming ASR model receives the audio stream
directly.

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

- top-bar status indicator for idle/listening/permission/error states
- small live transcript overlay for partial text
- committed phrase feedback after endpoint insertion

Partial text is never typed into the target app in v1.

The current overlay backend uses libadwaita when available, with a GTK 4
fallback. A GNOME Shell top-bar indicator remains future work.

## MVP Milestones

1. Runtime capability probe
   - GNOME version
   - portal availability
   - GlobalShortcuts availability
   - RemoteDesktop keyboard availability
   - EIS/libei availability
   - status: initial probe implemented

2. Streaming ASR spike
   - load Nemotron int8 model
   - stream microphone audio
   - print partial and endpoint-committed text
   - log real-time factor, queue depth, and commit latency
   - status: ASR worker protocol implemented; needs sherpa/model runtime test

3. ASR process protocol
   - newline-delimited JSON over stdio or Unix socket
   - events: `partial`, `commit`, `error`, `ready`
   - commands: `start`, `stop`, `shutdown`
   - status: implemented over stdio

4. Keyboard insertion spike
   - portal permission flow
   - insert simple ASCII phrases into focused GNOME apps
   - test Text Editor, Terminal, Firefox, LibreOffice
   - status: RemoteDesktop `NotifyKeyboardKeysym` backend implemented; needs
     live portal test outside sandbox

5. GNOME hotkey and status
   - global shortcut or shell extension trigger
   - daemon session state
   - visible listening/error state
   - status: GlobalShortcuts daemon path implemented; visual status not
     implemented

6. First integrated dictation
   - hotkey controls dictation
   - overlay shows partials
   - endpoint commits insert text
   - stop commits non-empty partial
   - status: CLI daemon, hotkey daemon, and optional Adwaita/GTK overlay
     implemented; top-bar indicator not implemented

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
- Model installation: manually configured model directory or managed downloader.
- Whether v1 supports spoken punctuation commands or inserts raw ASR output.
