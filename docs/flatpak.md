# Flatpak Packaging

Wordpipe is being packaged as an app-first Flatpak with a GNOME/libadwaita UI.
The Flatpak app owns the ASR daemon, model profiles, microphone capture, and
portal-based text insertion. A GNOME Shell extension can remain optional later.

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
Cargo crates during the Rust worker build and downloads ONNX Runtime from the
official release archive. Before a reproducible/distributable Flatpak build,
generate and add:

- `packaging/flatpak/cargo-sources.json` from `Cargo.lock`

The Python app uses PyGObject/GIO from the GNOME runtime for portal D-Bus access,
so `dbus-python` is not required. ONNX Runtime is installed from the official
Linux x64 archive as `/app/lib/libonnxruntime.so`.

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

Run the app:

```sh
flatpak run dev.wordpipe.Wordpipe
```

Command-line diagnostics remain available through the same Flatpak command:

```sh
flatpak run dev.wordpipe.Wordpipe probe
flatpak run dev.wordpipe.Wordpipe model-profiles
flatpak run dev.wordpipe.Wordpipe voice-keyboard --model-profile compact
```

## First-Run Model Flow

The GUI now opens even when the selected model profile is missing, showing setup
state instead of exiting before GTK starts. The app Flatpak intentionally does
not ship the full NeMo/PyTorch export stack. Instead, install a model profile by
importing a built Wordpipe runtime directory or archive into the Flatpak app data
directory.

For example, import a host-built compact profile:

```sh
flatpak run \
  --filesystem=/home/dhansen/.local/share/wordpipe/models:ro \
  dev.wordpipe.Wordpipe model-install \
  --profile compact \
  --source /home/dhansen/.local/share/wordpipe/models/nemotron-wordpipe-compact-fixed-shape-ort-format
```

After that copy, the model lives under
`~/.var/app/dev.wordpipe.Wordpipe/data/wordpipe/models/` and the runtime command
does not need the extra host filesystem grant:

```sh
flatpak run dev.wordpipe.Wordpipe app --model-profile compact
```

`model-install --source` also accepts a `.tar`, `.tar.gz`, `.tgz`, or `.zip`
archive containing a built Wordpipe profile directory with `tokenizer.model`,
`encoder.onnx` or `encoder.ort`, and `decoder_joint.onnx` or
`decoder_joint.ort`.

The non-Flatpak development environment can still build profiles directly from a
local `.nemo` file or Hugging Face NeMo repo id. Packaging that export toolchain
inside the app Flatpak is deferred because it requires NeMo, PyTorch, ONNX
tooling, and substantially more memory than the runtime.
