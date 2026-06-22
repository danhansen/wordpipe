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
