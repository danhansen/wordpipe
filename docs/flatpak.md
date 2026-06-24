# Flatpak Packaging

Wordpipe is being packaged as an app-first Flatpak with a GNOME/libadwaita UI.
The Flatpak app owns the ASR daemon, model profiles, microphone capture, and
portal-based text insertion. A GNOME Shell extension can remain optional later.

## Files

- `packaging/flatpak/dev.wordpipe.Wordpipe.yml`: Flatpak manifest.
- `packaging/flatpak/wordpipe-flatpak-launch`: launcher that defaults to
  `wordpipe app` and sets `ORT_DYLIB_PATH` when ONNX Runtime is installed under
  `/app`.
- `packaging/flatpak/requirements-runtime.txt`: Python runtime dependencies
  that need Flatpak source generation.
- `packaging/applications/dev.wordpipe.Wordpipe.desktop`: desktop entry.
- `packaging/metainfo/dev.wordpipe.Wordpipe.metainfo.xml`: AppStream metadata.
- `packaging/icons/hicolor/scalable/apps/dev.wordpipe.Wordpipe.svg`: app icon.

## Current Status

The manifest is a source-tree packaging skeleton. It is intentionally checked in
before vendored dependency source files so the install layout and permissions are
reviewable. Before a reproducible Flatpak build, generate and add:

- `packaging/flatpak/cargo-sources.json` from `Cargo.lock`
- `packaging/flatpak/python3-runtime-deps.json` from
  `packaging/flatpak/requirements-runtime.txt`

The Rust worker is built offline in the manifest, so the Cargo source list is
required. The Python runtime dependency list is required for `dbus-python` and
the ONNX Runtime wheel that provides `libonnxruntime.so`.

## Target Local Build

Install the builder and GNOME runtime/SDK:

```sh
sudo dnf install flatpak-builder
flatpak install flathub org.gnome.Platform//50 org.gnome.Sdk//50
```

After dependency source manifests exist, build and install locally:

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
state instead of exiting before GTK starts. The intended Flatpak flow is:

1. Launch Wordpipe.
2. Pick `compact` or `fast`.
3. Download or reuse the source NeMo checkpoint.
4. Build the selected profile into the Flatpak app data directory.
5. Start dictation after the model profile is installed.

The command-line version of that flow is already:

```sh
flatpak run dev.wordpipe.Wordpipe model-install --profile compact
flatpak run dev.wordpipe.Wordpipe app --model-profile compact
```

The next UI task is adding model-profile selection and install controls to the
libadwaita window.
