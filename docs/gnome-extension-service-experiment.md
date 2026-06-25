# GNOME Extension Service Experiment

This branch explores replacing the current Python/libadwaita Flatpak-centered
runtime with a GNOME Shell extension backed by a long-lived Rust service.

The existing Flatpak app remains the reference/fallback path on `main`. This
experiment should keep the ASR/model work reusable for KDE and other Wayland
desktops by putting the dictation engine behind a session D-Bus API.

## Target Shape

```text
GNOME Shell extension
  - global shortcut
  - panel or quick settings status
  - preferences/setup UI
  - GNOME Shell OSK/internal text insertion
  - D-Bus client

Rust wordpipe service
  - D-Bus server
  - model profile install/export/runtime
  - microphone/device enumeration
  - audio capture
  - ASR streaming
  - session state
  - append-only transcript events
```

The GNOME extension owns desktop integration. The Rust service owns work and
state. D-Bus is the boundary.

## Packaging Model

For local development, one installer can install both pieces:

```text
scripts/install-wordpipe-gnome
  - build/install Rust service under ~/.local/libexec/wordpipe/
  - install D-Bus service activation file
  - optionally install a systemd --user unit
  - install GNOME Shell extension under ~/.local/share/gnome-shell/extensions/
  - compile schemas
  - enable the extension
  - open extension preferences
```

For `extensions.gnome.org`, the extension likely cannot bundle the Rust binary.
GNOME extension review guidelines prohibit shipping binary executables or
libraries inside reviewed extensions. The extension should therefore be able to
detect and explain a missing `wordpipe-service`.

## D-Bus API Draft

Well-known name:

```text
dev.wordpipe.Service
```

Object path:

```text
/dev/wordpipe/Service
```

Interface:

```text
dev.wordpipe.Service1
```

Methods:

```text
Start() -> ()
Stop() -> ()
Toggle() -> ()
GetState() -> a{sv}
ListInputDevices() -> aa{sv}
SetInputDevice(s selector) -> ()
SetModelProfile(s profile) -> ()
InstallModel(s profile) -> ()
SetInsertionOptions(a{sv} options) -> ()
```

Signals:

```text
StateChanged(a{sv} state)
SessionStarted(t session_id)
TextDelta(t session_id, t seq, s text)
Partial(t session_id, t seq, s full_text)
Commit(t session_id, t seq, s text)
SessionStopped(t session_id)
InstallProgress(s profile, a{sv} progress)
Metrics(a{sv} metrics)
Error(s message)
```

`TextDelta` is append-only and is the primary signal for insertion clients.
`Partial` is for display/debugging. `session_id` and `seq` let clients ignore
stale events after toggles, restarts, or extension reloads.

## Migration Steps

1. Add a Rust crate for the service protocol and D-Bus server.
2. Move the current Rust worker loop into long-lived service state.
3. Preserve the current JSON-line worker mode for benchmarks during migration.
4. Add a minimal GNOME Shell extension that can connect to D-Bus and show
   service state.
5. Add extension preferences for model profile, mic, and insertion options.
6. Replace portal keyboard insertion with GNOME Shell insertion adapter.
7. Add local installer script for the service plus extension.
8. Keep KDE/other desktop clients as separate adapters using the same D-Bus API.

## Open Questions

- Which GNOME Shell versions should the first extension target? The current dev
  machine is expected to be GNOME 50-era, but the extension should not claim
  unsupported future versions.
- Which GNOME Shell internal OSK API is stable enough for a first experiment?
- Should model export/install live entirely in the Rust service, or should the
  first service call existing Python export scripts as a temporary bridge?
- Should the service be D-Bus activated only, systemd-user managed only, or
  support both?
