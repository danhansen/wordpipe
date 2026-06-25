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
Shutdown() -> ()
GetState() -> a{sv}
GetConfig() -> a{sv}
ListBackends() -> aa{sv}
ListModelProfiles() -> aa{sv}
ListInputDevices() -> aa{sv}
SetBackend(s backend) -> ()
SetInputDevice(s selector) -> ()
SetModelProfile(s profile) -> ()
SetShortcut(s accelerator) -> ()
InstallModel(s profile) -> ()
SetInsertionOptions(a{sv} options) -> ()
SetRuntimeOptions(a{sv} options) -> ()
```

Signals:

```text
StateChanged(a{sv} state)
ConfigChanged(a{sv} config)
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

`GetState` includes `selected_runtime_dir` and `selected_model_installed`, so
clients can disable start controls and steer users to model setup before the
service attempts to spawn the worker.
The service applies spoken-punctuation normalization before computing
`TextDelta` when `spoken_punctuation` is enabled. Partial normalization holds
ambiguous trailing command prefixes such as `new`, `question`, `full`, and
`exclamation` until they either become a complete command or ordinary text.
The GNOME client inserts `TextDelta` only when `insert_partials` is enabled and
inserts `Commit` text when it adds text that has not already been streamed.
`Stop()` moves state to `stopping=true` while the worker flushes. The service
emits final `Commit` before `SessionStopped` when the worker produces one.

## Migration Steps

1. Add a Rust crate for the service protocol and D-Bus server. Done.
2. Move the current Rust worker loop into long-lived service state. Next.
3. Preserve the current JSON-line worker mode for benchmarks during migration.
   Done.
4. Add a minimal GNOME Shell extension that can connect to D-Bus and show
   service state. Done.
5. Add extension preferences for model profile, mic, and insertion options.
   Done.
6. Replace portal keyboard insertion with GNOME Shell insertion adapter. Done
   for append-only `TextDelta` commits; live cross-app validation remains.
7. Add local installer script for the service plus extension. Done.
8. Keep KDE/other desktop clients as separate adapters using the same D-Bus API.
   Ongoing.

## Current Experiment State

The branch now has three Rust workspace crates:

```text
crates/wordpipe-protocol
  Shared D-Bus constants and model/backend profile metadata.

crates/wordpipe-service
  Session D-Bus service. It owns config/state, lists CPAL input devices,
  exposes model/backend setup methods, supervises the Rust ASR worker, and
  translates worker JSON events into the planned streaming signals.

crates/wordpipe-parakeet-worker
  Existing JSON-line ASR worker retained for benchmarks and as the code source
  for the upcoming service runtime migration.
```

The GNOME Shell extension lives at:

```text
extensions/gnome-shell/wordpipe@dhansen.dev
```

It currently provides:

- Panel indicator and menu for start/stop, service-provided model profile
  selection, and missing-profile install actions.
- GNOME Shell global shortcut using the extension's GSettings key.
- Preferences UI for service-provided backend/model profile lists,
  microphone selection, streaming insertion options, runtime options, overlay,
  shortcut, service status, and model setup progress.
- Overlay/status updates driven by D-Bus state and transcript signals.
- Config synchronization from `ConfigChanged`, so preferences and the shell
  client stay aligned with service-side changes.

The `TextInjector` in `extension.js` is intentionally isolated. The first
GNOME Shell 50 implementation commits append-only text deltas through
`Clutter.get_default_backend().get_input_method().commit(text)`. This is an
internal Shell/input-method path, so it needs live GNOME testing across focused
GTK/libadwaita apps, terminals, and browser text fields.

## Local Install

Before installing, verify that the GJS clients still match the Rust D-Bus
protocol declaration:

```bash
python3 scripts/check_gnome_dbus_xml.py
```

Install the service and extension for the current user:

```bash
scripts/install-wordpipe-gnome
```

The installer:

- Builds `wordpipe-service` in release mode.
- Builds `wordpipe-parakeet-worker` in release mode.
- Installs both Rust binaries to `~/.local/libexec/wordpipe/`.
- Installs a `wordpipe-model-install` wrapper next to the service. The wrapper
  calls the Python model installer from this checkout for local development.
- Installs a session D-Bus activation file for `dev.wordpipe.Service`.
- Installs a systemd user unit at
  `~/.config/systemd/user/wordpipe-service.service`.
- Copies the extension to
  `~/.local/share/gnome-shell/extensions/wordpipe@dhansen.dev`.
- Runs `glib-compile-schemas` for the installed extension.
- Enables the extension with `gnome-extensions enable`.

Open preferences after install:

```bash
gnome-extensions prefs wordpipe@dhansen.dev
```

In the current GNOME Shell session, newly copied extensions may not be
discovered until logging out and back in. If `gnome-extensions info
wordpipe@dhansen.dev` reports that the extension does not exist immediately
after install, log out/in and then run:

```bash
gnome-extensions enable wordpipe@dhansen.dev
```

Start the service explicitly for development:

```bash
systemctl --user start wordpipe-service.service
```

Or let D-Bus activate it when the extension first calls the service.

## Service Configuration

The Rust service persists user-facing configuration in:

```text
~/.config/wordpipe/service.json
```

Use `wordpipe-service --config /path/to/service.json` for isolated testing.
When no saved `model_profile` exists yet, the service keeps the default profile
only if it is installed; otherwise it selects the first installed profile it can
find under `model_root`. An explicitly saved valid profile remains authoritative
even when its model files are not installed yet, so the UI can still drive that
profile's install flow.
The GNOME extension mirrors service config on startup before pushing GSettings
changes back, so an extension reload should not overwrite the service's saved
model profile, microphone, shortcut, model root, sample rate, thread count, or
insertion options with schema defaults.

Runtime options that affect the worker process are updated through:

```text
SetRuntimeOptions(a{sv})
```

Supported keys are `model_root`, `worker_path`, `model_installer_path`,
`sample_rate`, and `num_threads`. Changes that affect the active worker stop the
current worker so the next dictation session starts with the new runtime
settings.

## Open Questions

- Which GNOME Shell versions should the first extension target? The current dev
  machine is expected to be GNOME 50-era, but the extension should not claim
  unsupported future versions.
- Which GNOME Shell internal OSK API is stable enough for a first experiment?
- Should release packaging keep the Python model installer as the ORT
  conversion bridge, or should download/extract/ORT conversion move into Rust?
- Should the service be D-Bus activated only, systemd-user managed only, or
  support both?
