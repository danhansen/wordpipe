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

Successful builds remove intermediate export files under
`~/.var/app/dev.wordpipe.Wordpipe/data/wordpipe/models/build/` by default. Pass
`--keep-build-dir` to `model-install` when debugging the export pipeline.

`model-install --source` also accepts a `.tar`, `.tar.gz`, `.tgz`, or `.zip`
archive containing a built Wordpipe profile directory with `tokenizer.model`,
`encoder.onnx` or `encoder.ort`, and `decoder_joint.onnx` or
`decoder_joint.ort`.

The non-Flatpak development environment can still build profiles directly from a
local `.nemo` file or Hugging Face NeMo repo id.
