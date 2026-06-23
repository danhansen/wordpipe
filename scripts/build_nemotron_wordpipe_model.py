#!/usr/bin/env python3
"""Build the current best Wordpipe Nemotron model from a NeMo checkpoint.

This is intentionally a thin orchestration layer. The individual phase scripts
remain independently runnable and debuggable; this wrapper records the blessed
phase order and the reason each phase exists.

Pipeline:

1. Export FP32 ONNX from NeMo.
   NeMo/Torch owns this step. We stop after FP32 export so the large Torch model
   can exit before ONNX Runtime quantization starts, which avoids holding both
   memory-heavy stacks at once.

2. Quantize and rewrite projected K/V cache.
   Starting from FP32 lets ONNX Runtime apply one coherent dynamic QUInt8 pass.
   The projected-cache rewrite stores already-projected attention K/V tensors so
   the runtime avoids recomputing old context every streaming chunk.

3. Specialize fixed streaming shapes.
   Wordpipe currently runs the Nemotron c56 streaming shape: 65 mel frames in,
   7 encoder frames out, 56 projected-cache frames. Fixing these shapes removes
   symbolic shape plumbing and lets ORT fold more graph work.

4. Dequantize feed-forward MatMul/Gemm blocks back to FP32.
   Benchmarks on the Ivy Bridge test machine show ORT's FP32 GEMM path is faster
   than dynamic activation quantization plus int8 matmul overhead for these FFN
   blocks. This is the current best speed artifact despite the larger model.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = ROOT / "build" / "nemotron-wordpipe-pipeline"
PHASES = ("export", "transform", "fixed-shape", "ffn-fp32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help=".nemo path or NeMo/Hugging Face model id.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Final Wordpipe model directory. Contains encoder.onnx, decoder_joint.onnx, tokenizer.model.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help=f"Intermediate phase directory. Default: {DEFAULT_WORK_DIR}",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--force", action="store_true", help="Delete existing phase/output dirs before writing.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--start-at",
        choices=PHASES,
        default="export",
        help="Resume from a later phase when earlier outputs already exist.",
    )
    parser.add_argument(
        "--stop-after",
        choices=PHASES,
        default="ffn-fp32",
        help="Stop after this phase.",
    )
    parser.add_argument("--left-context", type=int, default=56)
    parser.add_argument("--right-context", type=int, default=6)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--verify-lang", default="en-US")
    parser.add_argument("--input-frames", type=int, default=65)
    parser.add_argument("--output-frames", type=int, default=7)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--cache-len", type=int, default=56)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--conv-context", type=int, default=8)
    parser.add_argument(
        "--ort-optimize-final",
        choices=("disable", "basic", "extended", "all"),
        default="extended",
        help="ORT optimization level to serialize after fixed-shape and FFN-FP32 phases.",
    )
    parser.add_argument("--ort-optimize-threads", type=int, default=1)
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Keep FP32 export artifacts after the transform phase.",
    )
    return parser.parse_args()


def phase_enabled(args: argparse.Namespace, phase: str) -> bool:
    start = PHASES.index(args.start_at)
    stop = PHASES.index(args.stop_after)
    index = PHASES.index(phase)
    if start > stop:
        raise SystemExit(f"--start-at {args.start_at!r} comes after --stop-after {args.stop_after!r}")
    return start <= index <= stop


def phase_output(args: argparse.Namespace, phase: str) -> Path:
    if phase in {"export", "transform"}:
        return args.work_dir / "01-fp32-export"
    if phase == "fixed-shape":
        return args.work_dir / "02-fixed-shape"
    if phase == "ffn-fp32":
        return args.output_dir
    raise ValueError(phase)


def prepare_output(path: Path, *, force: bool, dry_run: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"Refusing to overwrite existing path without --force: {path}")
        print(f"[pipeline] removing {path}")
        if not dry_run:
            shutil.rmtree(path)
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)


def run(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"[pipeline] {printable}", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    python = str(args.python)
    export_dir = phase_output(args, "export")
    fixed_dir = phase_output(args, "fixed-shape")

    if phase_enabled(args, "export"):
        prepare_output(export_dir, force=args.force, dry_run=args.dry_run)
        run(
            [
                python,
                "scripts/export_nemotron_parakeet_optimized.py",
                args.input,
                str(export_dir),
                "--left-context",
                str(args.left_context),
                "--right-context",
                str(args.right_context),
                "--sample-rate",
                str(args.sample_rate),
                "--verify-lang",
                args.verify_lang,
                "--export-only",
            ],
            dry_run=args.dry_run,
        )

    if phase_enabled(args, "transform"):
        run(
            [
                python,
                "scripts/transform_nemotron_parakeet_export.py",
                str(export_dir),
                "--quantize",
                "--projected-cache",
                *(["--keep-fp32"] if args.keep_fp32 else []),
            ],
            dry_run=args.dry_run,
        )

    if phase_enabled(args, "fixed-shape"):
        prepare_output(fixed_dir, force=args.force, dry_run=args.dry_run)
        run(
            [
                python,
                "scripts/build_nemotron_fixed_shape_model.py",
                "--source-dir",
                str(export_dir),
                "--output-dir",
                str(fixed_dir),
                "--input-frames",
                str(args.input_frames),
                "--output-frames",
                str(args.output_frames),
                "--num-layers",
                str(args.num_layers),
                "--cache-len",
                str(args.cache_len),
                "--hidden-dim",
                str(args.hidden_dim),
                "--conv-context",
                str(args.conv_context),
                "--ort-optimize-final",
                args.ort_optimize_final,
                "--ort-optimize-threads",
                str(args.ort_optimize_threads),
            ],
            dry_run=args.dry_run,
        )

    if phase_enabled(args, "ffn-fp32"):
        prepare_output(args.output_dir, force=args.force, dry_run=args.dry_run)
        run(
            [
                python,
                "scripts/dequantize_nemotron_matmul_blocks.py",
                "--source-dir",
                str(fixed_dir),
                "--output-dir",
                str(args.output_dir),
                "--include",
                "/feed_forward",
                "--ort-optimize-final",
                args.ort_optimize_final,
                "--ort-optimize-threads",
                str(args.ort_optimize_threads),
            ],
            dry_run=args.dry_run,
        )

    if args.stop_after == "ffn-fp32":
        print(f"[pipeline] complete final={args.output_dir}", flush=True)
    else:
        print(f"[pipeline] stopped after {args.stop_after} output={phase_output(args, args.stop_after)}", flush=True)


if __name__ == "__main__":
    main()
