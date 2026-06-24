#!/usr/bin/env python3
"""Slim a Wordpipe Nemotron ONNX model directory with onnxslim.

This is an experimental preprocessing step before ORT-format conversion. The
defaults are intentionally conservative for large external-data models: skip
shape inference and avoid folding very large constants, while still allowing
dead-node elimination, graph cleanup, and small constant folding.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


GRAPH_FILES = ("encoder.onnx", "decoder_joint.onnx")
SUPPORT_FILES = ("config.json", "tokenizer.model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--force", action="store_true", help="Overwrite the output directory.")
    parser.add_argument(
        "--shape-infer",
        action="store_true",
        help="Enable onnxslim shape inference. Disabled by default to reduce peak memory.",
    )
    parser.add_argument(
        "--size-threshold",
        type=int,
        default=1_048_576,
        help="Maximum constant size, in bytes, that onnxslim may fold. Default: 1 MiB.",
    )
    parser.add_argument(
        "--skip-constant-folding",
        action="store_true",
        help="Disable constant folding entirely.",
    )
    parser.add_argument(
        "--skip-graph-fusion",
        action="store_true",
        help="Disable onnxslim graph fusion.",
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Only slim encoder.onnx and copy decoder_joint.onnx unchanged.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")


def prepare_output(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"Refusing to overwrite existing path without --force: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_support_files(source_dir: Path, output_dir: Path) -> None:
    for name in SUPPORT_FILES:
        source_path = source_dir / name
        require_file(source_path)
        shutil.copy2(source_path, output_dir / name)


def copy_graph(source_dir: Path, output_dir: Path, name: str) -> None:
    source_path = source_dir / name
    require_file(source_path)
    shutil.copy2(source_path, output_dir / name)
    external_data = source_path.with_name(f"{name}.data")
    if external_data.exists():
        shutil.copy2(external_data, output_dir / external_data.name)


def validate_source_dir(source_dir: Path) -> None:
    if not source_dir.is_dir():
        raise SystemExit(f"Source is not a directory: {source_dir}")
    for name in (*GRAPH_FILES, *SUPPORT_FILES):
        require_file(source_dir / name)


def slim_graph(args: argparse.Namespace, source_dir: Path, output_dir: Path, name: str) -> None:
    source_path = source_dir / name
    output_path = output_dir / name
    require_file(source_path)
    stage_dir = Path(
        tempfile.mkdtemp(prefix="wordpipe-onnxslim-input-", dir=output_dir.parent)
    )
    staged_source_path = stage_dir / name
    shutil.copy2(source_path, staged_source_path)
    external_data = source_path.with_name(f"{name}.data")
    if external_data.exists():
        shutil.copy2(external_data, stage_dir / external_data.name)

    skip_optimizations = []
    if args.skip_constant_folding:
        skip_optimizations.append("constant_folding")
    if args.skip_graph_fusion:
        skip_optimizations.append("graph_fusion")

    command = [
        str(args.python),
        "-m",
        "onnxslim",
        str(staged_source_path),
        str(output_path),
        "--save-as-external-data",
        "--size-threshold",
        str(args.size_threshold),
    ]
    if not args.shape_infer:
        command.append("--no-shape-infer")
    if skip_optimizations:
        command.extend(["--skip-optimizations", *skip_optimizations])

    print(f"[onnxslim] {' '.join(command)}", flush=True)
    try:
        subprocess.run(command, check=True)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    validate_source_dir(source_dir)
    prepare_output(output_dir, force=args.force)
    copy_support_files(source_dir, output_dir)

    slim_graph(args, source_dir, output_dir, "encoder.onnx")
    if args.encoder_only:
        copy_graph(source_dir, output_dir, "decoder_joint.onnx")
    else:
        slim_graph(args, source_dir, output_dir, "decoder_joint.onnx")

    print(f"[onnxslim] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
