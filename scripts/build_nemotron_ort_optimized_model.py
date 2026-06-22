#!/usr/bin/env python3
"""Build a model directory whose encoder is ORT's serialized optimized graph."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnxruntime as ort


ORT_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def build_variant(source_dir: Path, output_dir: Path, level: str, threads: int) -> None:
    source_encoder = source_dir / "encoder.onnx"
    if not source_encoder.exists():
        raise SystemExit(f"Missing source encoder: {source_encoder}")
    output_dir.mkdir(parents=True, exist_ok=True)

    options = ort.SessionOptions()
    options.optimized_model_filepath = str(output_dir / "encoder.onnx")
    options.graph_optimization_level = ORT_LEVELS[level]
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    ort.InferenceSession(str(source_encoder), sess_options=options, providers=["CPUExecutionProvider"])

    for sidecar in ("decoder_joint.onnx", "tokenizer.model", "config.json"):
        src = source_dir / sidecar
        if src.exists():
            hardlink_or_copy(src, output_dir / sidecar)

    print(f"[ort-opt] wrote {output_dir} level={level}")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size / 1024 / 1024:.1f} MiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--level", choices=sorted(ORT_LEVELS), default="extended")
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_variant(args.source_dir, args.output_dir, args.level, args.threads)


if __name__ == "__main__":
    main()
