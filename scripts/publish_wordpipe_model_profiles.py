#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wordpipe.models import (  # noqa: E402
    DEFAULT_PREBUILT_PROFILE_REPO,
    MODEL_PROFILES,
    ModelProfileSpec,
    model_runtime_dir_valid,
    profile_spec,
)


DEFAULT_OUTPUT_DIR = ROOT / "build" / "model-release"
REQUIRED_ONNX_FILES = ("tokenizer.model", "encoder.onnx", "decoder_joint.onnx")
OPTIONAL_PROFILE_FILES = (
    "encoder.onnx.data",
    "decoder_joint.onnx.data",
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)


def main() -> int:
    args = parse_args()
    profiles = tuple(dict.fromkeys(args.profile or ("fast", "compact")))
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    archives: list[Path] = []
    manifest: dict[str, object] = {
        "repo_id": args.repo_id,
        "source_model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "profiles": {},
    }

    for profile_name in profiles:
        spec = profile_spec(profile_name)
        source = resolve_profile_source(args, spec)
        validate_publish_source(source)
        archive = output_dir / spec.prebuilt_filename
        package_profile(
            source,
            archive,
            spec=spec,
            force=args.force,
            compresslevel=args.compresslevel,
        )
        archives.append(archive)
        manifest["profiles"][profile_name] = archive_metadata(archive, source, spec)
        print(f"{profile_name}: {archive}")

    manifest_path = output_dir / "wordpipe-model-profiles-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")

    readme_path = output_dir / "README.md"
    if not readme_path.exists() or args.force_card:
        readme_path.write_text(render_model_card(args.repo_id, profiles), encoding="utf-8")
        print(f"model card: {readme_path}")

    if args.upload:
        upload_release(
            repo_id=args.repo_id,
            folder_path=output_dir,
            private=args.private,
            revision=args.revision,
            commit_message=args.commit_message,
        )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package and optionally upload Wordpipe-specialized Nemotron model profiles "
            "to the Hugging Face Hub."
        )
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=tuple(MODEL_PROFILES),
        help="Profile to package. May be passed more than once. Defaults to fast and compact.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        help="Directory containing profile output directories using Wordpipe's canonical names.",
    )
    parser.add_argument("--fast-dir", type=Path, help="Built ONNX directory for the fast profile.")
    parser.add_argument("--compact-dir", type=Path, help="Built ONNX directory for the compact profile.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for release archives and README.md. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_PREBUILT_PROFILE_REPO,
        help="Hugging Face model repo id to publish to.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing local archives.")
    parser.add_argument(
        "--compresslevel",
        type=int,
        default=1,
        choices=range(0, 10),
        metavar="0..9",
        help="gzip compression level for tar.gz archives. Defaults to 1 for release speed.",
    )
    parser.add_argument(
        "--force-card",
        action="store_true",
        help="Overwrite an existing generated README.md in the output directory.",
    )
    parser.add_argument("--upload", action="store_true", help="Upload output-dir to Hugging Face.")
    parser.add_argument("--private", action="store_true", help="Create the Hugging Face repo as private.")
    parser.add_argument("--revision", help="Branch, tag, or PR ref to upload to.")
    parser.add_argument(
        "--commit-message",
        default="Publish Wordpipe Nemotron model profiles",
        help="Commit message for Hugging Face upload.",
    )
    return parser.parse_args()


def resolve_profile_source(args: argparse.Namespace, spec: ModelProfileSpec) -> Path:
    override = getattr(args, f"{spec.name}_dir")
    if override is not None:
        return override.expanduser()
    if args.model_root is not None:
        return spec.output_dir(args.model_root.expanduser())
    raise SystemExit(f"{spec.name}: pass --{spec.name}-dir or --model-root")


def validate_publish_source(source: Path) -> None:
    source = source.expanduser()
    if not source.is_dir():
        raise SystemExit(f"{source} is not a directory")
    if not model_runtime_dir_valid(source):
        raise SystemExit(
            f"{source} is not a built Wordpipe profile; expected tokenizer.model plus encoder/decoder files"
        )
    missing = [name for name in REQUIRED_ONNX_FILES if not (source / name).is_file()]
    if missing:
        raise SystemExit(
            f"{source} is not publishable as a prebuilt profile archive; missing {', '.join(missing)}. "
            "Publish the ONNX profile directory, not the local ORT runtime cache."
        )


def package_profile(
    source: Path,
    archive: Path,
    *,
    spec: ModelProfileSpec,
    force: bool,
    compresslevel: int = 1,
) -> None:
    if archive.exists() and not force:
        raise SystemExit(f"{archive} already exists; pass --force to overwrite it")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{archive.name}.",
        suffix=".tmp",
        dir=archive.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=compresslevel) as tar:
            for path in publish_files(source.expanduser()):
                tar.add(path, arcname=f"{spec.output_name}/{path.name}", filter=normalize_tar_member)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_tar_member(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if not (info.isfile() or info.isdir()):
        return None
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    if info.isdir():
        info.mode = 0o755
    else:
        info.mode = 0o644
    return info


def publish_files(source: Path) -> list[Path]:
    names = [*REQUIRED_ONNX_FILES, *OPTIONAL_PROFILE_FILES]
    return [source / name for name in names if (source / name).is_file()]


def archive_metadata(archive: Path, source: Path, spec: ModelProfileSpec) -> dict[str, object]:
    return {
        "title": spec.title,
        "description": spec.description,
        "build_profile": spec.build_profile,
        "archive": archive.name,
        "source_dir": str(source),
        "sha256": sha256_file(archive),
        "bytes": archive.stat().st_size,
        "contains": [path.name for path in publish_files(source)],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_model_card(repo_id: str, profiles: Iterable[str]) -> str:
    profile_lines = []
    for profile_name in profiles:
        spec = profile_spec(profile_name)
        profile_lines.append(
            f"- `{spec.prebuilt_filename}`: {spec.title} profile. {spec.description}"
        )
    joined_profiles = "\n".join(profile_lines)
    return f"""---
language:
- en
tags:
- automatic-speech-recognition
- onnx
- wordpipe
- nemotron
library_name: onnx
base_model: nvidia/nemotron-3.5-asr-streaming-0.6b
---

# Wordpipe Nemotron 3.5 ASR Streaming Profiles

This repository contains Wordpipe-specialized ONNX profile archives derived from
`nvidia/nemotron-3.5-asr-streaming-0.6b`.

The archives are consumed by Wordpipe with:

```sh
wordpipe model-install --profile fast --prebuilt-repo {repo_id}
wordpipe model-install --profile compact --prebuilt-repo {repo_id}
```

## Files

{joined_profiles}

`wordpipe-model-profiles-manifest.json` records archive sizes, SHA-256 hashes,
source directories used when packaging, and the Wordpipe build profile.

## Notes

These are inference artifacts for local desktop dictation. They are not training
checkpoints. The `compact` archive intentionally contains the ONNX profile; the
Wordpipe installer converts it to ORT format locally for startup-time behavior.
"""


def upload_release(
    *,
    repo_id: str,
    folder_path: Path,
    private: bool,
    revision: str | None,
    commit_message: str,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for --upload. Install it, then run `hf auth login`."
        ) from exc

    try:
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        HfHubHTTPError = Exception

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            commit_message=commit_message,
            allow_patterns=[
                "README.md",
                "wordpipe-model-profiles-manifest.json",
                "*.tar.gz",
            ],
        )
    except HfHubHTTPError as exc:
        raise SystemExit(
            f"Failed to upload to {repo_id}. Make sure `hf auth login` is using a token "
            "with write access to this model repository."
        ) from exc
    print(f"uploaded: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    raise SystemExit(main())
