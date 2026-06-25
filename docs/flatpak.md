# Flatpak Packaging

Wordpipe is being packaged as an app-first Flatpak with a GNOME/libadwaita UI.
The Flatpak app owns the ASR daemon, model profiles, microphone capture, and
portal-based text insertion. A GNOME Shell extension can remain optional later.
The HIG-informed completion plan is in `docs/gui-flatpak-plan.md`.

## Files

- `packaging/flatpak/dev.wordpipe.Wordpipe.yml`: Flatpak manifest.
- `packaging/flatpak/wordpipe-flatpak-launch`: launcher that defaults to
  `wordpipe app` and sets `ORT_DYLIB_PATH` when ONNX Runtime is installed under
  `/app`.
- `packaging/flatpak/requirements-runtime.txt`: placeholder for future Python
  runtime dependencies that need Flatpak source generation.
- `packaging/applications/dev.wordpipe.Wordpipe.desktop`: desktop entry.
- `packaging/metainfo/dev.wordpipe.Wordpipe.metainfo.xml`: AppStream metadata.
- `packaging/icons/hicolor/scalable/apps/dev.wordpipe.Wordpipe.svg`: app icon.

## Current Status

The manifest is currently a network-enabled local development build. It fetches
Cargo crates during the Rust worker build, downloads ONNX Runtime from the
official release archive, and installs the Python model export stack with pip.
Before a reproducible/distributable Flatpak build, generate and add:

- `packaging/flatpak/cargo-sources.json` from `Cargo.lock`
- Flatpak source manifests for `packaging/flatpak/requirements-model-tools.txt`

The Python app uses PyGObject/GIO from the GNOME runtime for portal D-Bus access,
so `dbus-python` is not required. Runtime ONNX Runtime is installed from the
official Linux x64 archive as `/app/lib/libonnxruntime.so`; the Python
`onnxruntime` package is also installed for quantization/export tooling.

## Target Local Build

Install the builder and GNOME runtime/SDK:

```sh
sudo dnf install flatpak-builder
flatpak install flathub org.gnome.Platform//50 org.gnome.Sdk//50
```

Build and install locally:

```sh
flatpak-builder --user --install --force-clean \
  build/flatpak-dev \
  packaging/flatpak/dev.wordpipe.Wordpipe.yml
```

If the build fails with `rofiles-fuse` or `/dev/fuse` errors, use the same
command with `--disable-rofiles-fuse`:

```sh
flatpak-builder --disable-rofiles-fuse --user --install --force-clean \
  build/flatpak-dev \
  packaging/flatpak/dev.wordpipe.Wordpipe.yml
```

Run the app:

```sh
flatpak run dev.wordpipe.Wordpipe
```

For normal Python/UI iteration after one full Flatpak install, do not rebuild
the Flatpak. Run the installed sandbox with the source checkout mounted
read-only instead:

```sh
scripts/wordpipe-flatpak-dev app
scripts/wordpipe-flatpak-dev model-profiles
scripts/wordpipe-flatpak-dev voice-keyboard --model-profile compact
```

That path still uses the Flatpak runtime, app-data model directory, portals,
and packaged Rust worker, but imports `wordpipe` from `src/`. Rebuild the
Flatpak only when the manifest, desktop metadata, bundled scripts, Python
dependencies, Rust worker, or ONNX Runtime packaging changes. Local `type: dir`
sources are not module-cached by Flatpak Builder, so full local rebuilds can
still revisit the Rust and model-tools modules.

Command-line diagnostics remain available through the same Flatpak command:

```sh
flatpak run dev.wordpipe.Wordpipe probe
flatpak run dev.wordpipe.Wordpipe model-profiles
flatpak run dev.wordpipe.Wordpipe audio-devices --backend parakeet
flatpak run dev.wordpipe.Wordpipe voice-keyboard --model-profile compact
```

## First-Run Model Flow

The GUI now opens even when the selected model profile is missing, showing setup
state instead of exiting before GTK starts. The app Flatpak includes the NeMo,
PyTorch, Hugging Face, ONNX, and ONNX Runtime tooling needed to download and
convert a selected profile into the canonical Flatpak app-data model directory.

Open the app, choose `Fast` or `Compact`, and press `Install` to download the
source NeMo checkpoint and build that profile in the background:

```sh
flatpak run dev.wordpipe.Wordpipe
```

The same operation can be run from the command line, which is useful for
unattended setup or export debugging:

```sh
flatpak run dev.wordpipe.Wordpipe model-install --profile compact
```

After that build, the model lives under
`~/.var/app/dev.wordpipe.Wordpipe/data/wordpipe/models/`, and runtime commands
do not need model paths:

```sh
flatpak run dev.wordpipe.Wordpipe app --model-profile compact
```

For focus-preserving dictation into another app, install a host GNOME custom
shortcut that starts and toggles the resident Flatpak voice keyboard:

```sh
scripts/install-wordpipe-flatpak-gnome-shortcut
```

The helper wraps the same implementation exposed by the development CLI:

```sh
scripts/wordpipe-dev shortcut-status --target flatpak
scripts/wordpipe-dev shortcut-install --target flatpak
```

Then focus a text field and press `Ctrl+Alt+Space` to start or stop dictation.
The first press starts `voice-keyboard --signal-hotkey` inside the Flatpak when
needed, waits for the ready pid file, and then toggles dictation. For visible
logs while debugging, run the resident daemon manually before pressing the
shortcut:

```sh
flatpak run dev.wordpipe.Wordpipe voice-keyboard --signal-hotkey
```

If you need to select a non-default microphone, use the Parakeet device list
because it comes from the same Rust/CPAL worker that records audio. Pass the
listed `cpal:N` selector to `voice-keyboard`, `listen-test`, or the app:

```sh
flatpak run dev.wordpipe.Wordpipe audio-devices --backend parakeet
flatpak run dev.wordpipe.Wordpipe voice-keyboard --signal-hotkey --input-device cpal:0
```

The GUI exposes the same CPAL list in the Microphone row. Refresh the list,
choose a device, and Wordpipe writes the selected `cpal:N` selector to
`config.toml`; the app restarts its dictation controller against that input.
The Insertion rows persist live partial insertion and spoken-punctuation
settings the same way.

If the shortcut starts the daemon itself, startup logs are written inside the
Flatpak cache at
`~/.var/app/dev.wordpipe.Wordpipe/cache/wordpipe/voice-keyboard.log`.

The app profile selector writes `model_profile` into the Flatpak config, so the
resident daemon uses the selected profile unless `--model-profile` is passed.
The app also shows shortcut status and can try to repair it, but the host helper
above is the reliable path if the Flatpak sandbox cannot write GNOME's host
custom-keybinding settings.

Successful builds remove intermediate export files under
`~/.var/app/dev.wordpipe.Wordpipe/data/wordpipe/models/build/` by default. Pass
`--keep-build-dir` to `model-install` when debugging the export pipeline.

`model-install --source` also accepts a `.tar`, `.tar.gz`, `.tgz`, or `.zip`
archive containing a built Wordpipe profile directory with `tokenizer.model`,
`encoder.onnx` or `encoder.ort`, and `decoder_joint.onnx` or
`decoder_joint.ort`.

The non-Flatpak development environment can still build profiles directly from a
local `.nemo` file or Hugging Face NeMo repo id.
