#!/usr/bin/env python3
"""Build a Wordpipe Nemotron model from a NeMo checkpoint.

This is intentionally a thin orchestration layer. The individual phase scripts
remain independently runnable and debuggable; this wrapper records the blessed
phase order and the reason each phase exists.

Pipeline:

1. Export FP32 ONNX from NeMo.
   NeMo/Torch owns this step. We stop after FP32 export so the large Torch model
   can exit before ONNX Runtime quantization starts, which avoids holding both
   memory-heavy stacks at once.

2. Transform and rewrite projected K/V cache.
   The projected-cache rewrite stores already-projected attention K/V tensors so
   the runtime avoids recomputing old context every streaming chunk. The default
   high-performance profile keeps the encoder and decoder in FP32. The legacy
   compact and mixed profiles first run one coherent dynamic QUInt8 pass.

3. Specialize fixed streaming shapes.
   Wordpipe currently runs the Nemotron c56 streaming shape: 65 mel frames in,
   7 encoder frames out, 56 projected-cache frames. Fixing these shapes removes
   symbolic shape plumbing and lets ORT fold more graph work.

4. Optionally dequantize feed-forward MatMul/Gemm blocks back to FP32.
   Benchmarks on the Ivy Bridge test machine show ORT's FP32 GEMM path is faster
   than dynamic activation quantization plus int8 matmul overhead for these FFN
   blocks. This preserves the older mixed-int8/FP32 candidate.

5. Optionally convert the final ONNX components to native ORT format.
   ORT format avoids protobuf model parsing at runtime and lets the Rust worker
   use `session.use_ort_model_bytes_directly` for faster model startup. This is
   currently most useful for the compact profile; the FP32 encoder conversion is
   memory-heavy on 16 GB systems.
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
PROFILES = ("fp32-projected", "compact-fixed-shape", "ffn-fp32")


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
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="fp32-projected",
        help=(
            "Model build profile. fp32-projected is the current high-performance "
            "default; compact-fixed-shape keeps the small quantized/projected-cache "
            "encoder; ffn-fp32 adds FP32 feed-forward blocks to that compact base."
        ),
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
        default=None,
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
        "--constant-processed-signal-length",
        action="store_true",
        help=(
            "During the fixed-shape phase, replace processed_signal_length "
            "with a constant initializer. Experimental; benchmark before use."
        ),
    )
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
        help="Keep source FP32 export artifacts after the transform phase.",
    )
    parser.add_argument(
        "--projected-cache-current-projection",
        choices=("auto", "dynamic-int8", "fp32"),
        default="auto",
        help=(
            "Projection used for current K/V chunks in the projected-cache "
            "rewrite. auto preserves the default transform behavior."
        ),
    )
    parser.add_argument(
        "--quantize-per-channel",
        action="store_true",
        help="Experimental: use per-channel weights for dynamic QUInt8 quantization.",
    )
    parser.add_argument(
        "--fp32-decoder",
        action="store_true",
        help=(
            "Keep decoder_joint.onnx as FP32 while quantizing the encoder. "
            "Experimental speed/accuracy tradeoff; validate WER before use."
        ),
    )
    parser.add_argument(
        "--emit-ort-format",
        action="store_true",
        help="Also write a native ORT-format model directory after the final ONNX output is built.",
    )
    parser.add_argument(
        "--ort-format-output-dir",
        type=Path,
        help="Output directory for --emit-ort-format. Default: <output_dir>-ort-format.",
    )
    parser.add_argument(
        "--ort-format-optimization-level",
        choices=("disable", "basic", "extended", "all"),
        default="all",
        help="Optimization level used by convert_nemotron_to_ort_format.py.",
    )
    return parser.parse_args()


def phase_enabled(args: argparse.Namespace, phase: str) -> bool:
    phases = active_phases(args)
    start = PHASES.index(args.start_at)
    stop = PHASES.index(args.stop_after or phases[-1])
    index = PHASES.index(phase)
    if start > stop:
        raise SystemExit(f"--start-at {args.start_at!r} comes after --stop-after {args.stop_after!r}")
    return phase in phases and start <= index <= stop


def active_phases(args: argparse.Namespace) -> tuple[str, ...]:
    if args.profile in {"fp32-projected", "compact-fixed-shape"}:
        return ("export", "transform", "fixed-shape")
    if args.profile == "ffn-fp32":
        return PHASES
    raise ValueError(args.profile)


def phase_output(args: argparse.Namespace, phase: str) -> Path:
    if phase in {"export", "transform"}:
        return args.work_dir / "01-fp32-export"
    if phase == "fixed-shape":
        return args.output_dir if args.profile in {"fp32-projected", "compact-fixed-shape"} else args.work_dir / "02-fixed-shape"
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
    phases = active_phases(args)
    if args.stop_after is None:
        args.stop_after = phases[-1]
    if args.stop_after not in phases:
        raise SystemExit(f"--stop-after {args.stop_after!r} is not part of profile {args.profile!r}")
    if args.start_at not in phases:
        raise SystemExit(f"--start-at {args.start_at!r} is not part of profile {args.profile!r}")
    if args.profile != "ffn-fp32" and args.fp32_decoder:
        raise SystemExit("--fp32-decoder only applies to --profile ffn-fp32")
    if args.profile == "fp32-projected" and args.quantize_per_channel:
        raise SystemExit("--quantize-per-channel only applies to quantized profiles")

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
                "--no-quantize" if args.profile == "fp32-projected" else "--quantize",
                "--projected-cache",
                "--projected-cache-current-projection",
                args.projected_cache_current_projection,
                *(["--quantize-per-channel"] if args.profile != "fp32-projected" and args.quantize_per_channel else []),
                *(["--fp32-decoder"] if args.profile == "ffn-fp32" and args.fp32_decoder else []),
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
                *(["--constant-processed-signal-length"] if args.constant_processed_signal_length else []),
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

    if args.emit_ort_format and args.stop_after == phases[-1]:
        ort_format_dir = args.ort_format_output_dir or args.output_dir.with_name(
            f"{args.output_dir.name}-ort-format"
        )
        run(
            [
                python,
                "scripts/convert_nemotron_to_ort_format.py",
                str(args.output_dir),
                str(ort_format_dir),
                "--optimization-level",
                args.ort_format_optimization_level,
                *(["--force"] if args.force else []),
            ],
            dry_run=args.dry_run,
        )

    if args.stop_after == phases[-1]:
        print(f"[pipeline] complete final={args.output_dir}", flush=True)
    else:
        print(f"[pipeline] stopped after {args.stop_after} output={phase_output(args, args.stop_after)}", flush=True)


if __name__ == "__main__":
    main()
