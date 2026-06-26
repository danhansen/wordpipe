# Publishing Wordpipe Model Profiles

Wordpipe installs app-level model profiles from prebuilt Hugging Face archives by
default. The installer currently looks in:

```text
danhansen/wordpipe-nemotron-3.5-asr-streaming-0.6b
```

and expects these root-level files:

```text
wordpipe-nemotron-fast-fp32-projected.tar.gz
wordpipe-nemotron-compact-fixed-shape.tar.gz
```

The archives must contain ONNX profile directories with:

```text
tokenizer.model
encoder.onnx
decoder_joint.onnx
```

If an ONNX graph uses external tensor data, the matching `encoder.onnx.data` or
`decoder_joint.onnx.data` file must sit next to the graph. The publishing script
includes those sidecars automatically when present and excludes stale duplicate
export files such as `*.fp32.*`.

Do not publish the local `*-ort-format` runtime cache for `compact`; Wordpipe
downloads the compact ONNX archive and converts it to ORT format during install.

## Hub Conventions

The generated Hugging Face `README.md` is the model card. It uses YAML metadata
so the Hub can index the repository correctly:

- `pipeline_tag: automatic-speech-recognition`
- `library_name: onnx`
- `base_model: nvidia/nemotron-3.5-asr-streaming-0.6b`
- `license: openmdw-1.1`
- `language: multilingual`

The card should make clear that NVIDIA is the upstream model developer and that
Wordpipe publishes derived inference artifacts: export, graph specialization,
packaging, and desktop runtime integration. Do not claim new training datasets,
new upstream FLEURS results, or a separate checkpoint lineage unless those have
actually been produced and validated.

Before uploading, review the generated model card for:

- Attribution to `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- OpenMDW 1.1 license metadata and prose link.
- A clear intended-use section for local Wordpipe dictation.
- A limitations section explaining that `compact` is converted to ORT locally.
- Evaluation language that points to Wordpipe release documentation instead of
  copying NVIDIA's upstream benchmark numbers.

## Package Profiles

From canonical installed profile names under a model root:

```sh
PYTHONPATH=src python3 scripts/publish_wordpipe_model_profiles.py \
  --model-root ~/.local/share/wordpipe/models \
  --output-dir build/model-release \
  --force
```

Or from explicit build directories:

```sh
PYTHONPATH=src python3 scripts/publish_wordpipe_model_profiles.py \
  --fast-dir build/model-variants/nemotron-fp32-projected \
  --compact-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-release \
  --force
```

The script writes the two archives, a generated `README.md`, and
`wordpipe-model-profiles-manifest.json` with sizes and SHA-256 hashes.

## Upload To Hugging Face

Authenticate first:

```sh
hf auth login
```

Use a token with write access to the destination model repository. A read-only
token is sufficient for `model-install` downloads, but publishing needs write
permission because the script creates or updates the Hub repo.

Then upload:

```sh
PYTHONPATH=src python3 scripts/publish_wordpipe_model_profiles.py \
  --model-root ~/.local/share/wordpipe/models \
  --repo-id danhansen/wordpipe-nemotron-3.5-asr-streaming-0.6b \
  --output-dir build/model-release \
  --force \
  --upload
```

The script uses `huggingface_hub.HfApi.create_repo(..., exist_ok=True)` and
`upload_folder()` with model repo type. Recent Hugging Face tooling uses `hf_xet`
for large-file transfer; the script sets `HF_XET_HIGH_PERFORMANCE=1` by default
for uploads.

## Verify Install

After upload, verify both profiles from a clean model root:

```sh
PYTHONPATH=src python3 -m wordpipe model-install \
  --profile fast \
  --model-root /tmp/wordpipe-model-install-check \
  --force-source

PYTHONPATH=src python3 -m wordpipe model-install \
  --profile compact \
  --model-root /tmp/wordpipe-model-install-check \
  --force-source
```

`fast` should install directly as an ONNX runtime directory.
`compact` should install the ONNX profile and then produce the local
`nemotron-wordpipe-compact-fixed-shape-ort-format` runtime directory.
