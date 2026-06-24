#!/usr/bin/env python3
"""Convert a Wordpipe Nemotron model directory from ONNX to ORT format.

The runtime can load native `.ort` models with graph bytes used directly for
initializers, avoiding part of the protobuf parse/materialization cost of
standard `.onnx` files. This script keeps the model directory layout compatible
with Wordpipe by writing `encoder.ort`, `decoder_joint.ort`, `config.json`, and
`tokenizer.model` to the output directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from onnxruntime.tools.convert_onnx_models_to_ort import (
    OptimizationStyle,
    convert_onnx_models_to_ort,
)


GRAPH_FILES = ("encoder.onnx", "decoder_joint.onnx")
SUPPORT_FILES = ("config.json", "tokenizer.model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--optimization-level",
        choices=("disable", "basic", "extended", "all"),
        default="all",
        help="ORT graph optimization level used while creating .ort files.",
    )
    parser.add_argument(
        "--optimization-style",
        choices=("fixed", "runtime"),
        default="fixed",
        help="ORT format optimization style. fixed minimizes runtime startup work.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite the output directory.")
    parser.add_argument(
        "--save-optimized-onnx",
        action="store_true",
        help="Also emit optimized ONNX files beside the ORT files for inspection.",
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


def link_or_copy(source_path: Path, target_path: Path) -> None:
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def materialize_graph_input(source_dir: Path, work_dir: Path, name: str) -> Path:
    source_path = source_dir / name
    require_file(source_path)
    target_path = work_dir / name
    link_or_copy(source_path, target_path)
    external_data = source_path.with_name(f"{name}.data")
    if external_data.exists():
        link_or_copy(external_data, work_dir / external_data.name)
    return target_path


def copy_support_files(source_dir: Path, output_dir: Path) -> None:
    for name in SUPPORT_FILES:
        source_path = source_dir / name
        require_file(source_path)
        shutil.copy2(source_path, output_dir / name)


def validate_source_dir(source_dir: Path) -> None:
    if not source_dir.is_dir():
        raise SystemExit(f"Source is not a directory: {source_dir}")
    for name in (*GRAPH_FILES, *SUPPORT_FILES):
        require_file(source_dir / name)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    validate_source_dir(source_dir)
    prepare_output(output_dir, force=args.force)
    copy_support_files(source_dir, output_dir)

    style = {
        "fixed": OptimizationStyle.Fixed,
        "runtime": OptimizationStyle.Runtime,
    }[args.optimization_style]

    previous_level = os.environ.get("ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL")
    os.environ["ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL"] = args.optimization_level
    try:
        for graph_name in GRAPH_FILES:
            with tempfile.TemporaryDirectory(prefix="wordpipe-ort-convert-") as temp:
                work_dir = Path(temp)
                graph_path = materialize_graph_input(source_dir, work_dir, graph_name)
                print(f"[ort-format] converting {graph_name}", flush=True)
                convert_onnx_models_to_ort(
                    graph_path,
                    output_dir=output_dir,
                    optimization_styles=[style],
                    save_optimized_onnx_model=args.save_optimized_onnx,
                )
    finally:
        if previous_level is None:
            os.environ.pop("ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL", None)
        else:
            os.environ["ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL"] = previous_level

    print(f"[ort-format] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
