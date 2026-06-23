#!/usr/bin/env python3
"""Build and benchmark pending Sayboard-derived optimization experiments.

This script is intentionally an orchestration layer over the smaller phase
scripts. It keeps the remaining harvest experiments reproducible without
changing the current default model pipeline.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("fixed-length", "fp32-current-projection", "per-channel-quantization")


def run(command: list[str | Path], *, dry_run: bool) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"[harvest] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def prepare_dir(path: Path, *, force: bool, dry_run: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"Refusing to overwrite existing path without --force: {path}")
        print(f"[harvest] removing {path}", flush=True)
        if not dry_run:
            shutil.rmtree(path)
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def copy_or_reflink(src: Path, dst: Path, *, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = ["cp", "--reflink=auto", "--remove-destination", src, dst]
    run(command, dry_run=dry_run)


def benchmark(
    args: argparse.Namespace,
    *,
    label: str,
    model_dir: Path,
    output_json: Path,
) -> None:
    if args.skip_benchmark:
        return
    run(
        [
            args.python,
            "scripts/benchmark_parakeet_variant.py",
            f"baseline={args.baseline_model_dir}",
            f"{label}={model_dir}",
            "--runs",
            str(args.runs),
            "--num-threads",
            str(args.num_threads),
            "--min-mem-available-gb",
            str(args.min_mem_available_gb),
            "--child-memory-limit-gb",
            str(args.child_memory_limit_gb),
            "--set-power-profile",
            args.power_profile,
            "--output",
            output_json,
        ],
        dry_run=args.dry_run,
    )
    run(
        [
            args.python,
            "scripts/score_benchmark_wer.py",
            output_json,
        ],
        dry_run=args.dry_run,
    )


def build_worker(args: argparse.Namespace) -> None:
    if args.skip_worker_build or args.skip_benchmark:
        return
    run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "wordpipe-parakeet-worker",
        ],
        dry_run=args.dry_run,
    )


def build_fixed_length(args: argparse.Namespace) -> Path:
    fixed_dir = args.output_root / "fixed-length-fixed-shape"
    final_dir = args.output_root / "fixed-length-ffn-fp32-ort"
    prepare_dir(fixed_dir, force=args.force, dry_run=args.dry_run)
    prepare_dir(final_dir, force=args.force, dry_run=args.dry_run)
    run(
        [
            args.python,
            "scripts/build_nemotron_fixed_shape_model.py",
            "--source-dir",
            args.projected_int8_model_dir,
            "--output-dir",
            fixed_dir,
            "--constant-processed-signal-length",
            "--ort-optimize-final",
            args.ort_optimize_final,
            "--ort-optimize-threads",
            str(args.ort_optimize_threads),
        ],
        dry_run=args.dry_run,
    )
    run(
        [
            args.python,
            "scripts/dequantize_nemotron_matmul_blocks.py",
            "--source-dir",
            fixed_dir,
            "--output-dir",
            final_dir,
            "--include",
            "/feed_forward",
            "--ort-optimize-final",
            args.ort_optimize_final,
            "--ort-optimize-threads",
            str(args.ort_optimize_threads),
        ],
        dry_run=args.dry_run,
    )
    benchmark(
        args,
        label="fixed_length",
        model_dir=final_dir,
        output_json=args.bench_root / "sayboard-fixed-length-001.json",
    )
    return final_dir


def prepare_fp32_export(args: argparse.Namespace, export_dir: Path) -> None:
    prepare_dir(export_dir, force=args.force, dry_run=args.dry_run)
    required = [
        "encoder.fp32.consolidated.onnx",
        "encoder.fp32.consolidated.data",
        "decoder_joint.fp32.onnx",
        "tokenizer.model",
        "export_config.json",
    ]
    for name in required:
        src = args.fp32_export_dir / name
        if not src.exists():
            raise SystemExit(f"Missing FP32 export artifact: {src}")
        copy_or_reflink(src, export_dir / name, dry_run=args.dry_run)


def build_fp32_transform_variant(
    args: argparse.Namespace,
    *,
    stem: str,
    transform_extra_args: list[str],
    benchmark_label: str,
    output_json_name: str,
) -> Path:
    export_dir = args.output_root / f"{stem}-export"
    fixed_dir = args.output_root / f"{stem}-fixed-shape"
    final_dir = args.output_root / f"{stem}-ffn-fp32-ort"
    prepare_fp32_export(args, export_dir)
    prepare_dir(fixed_dir, force=args.force, dry_run=args.dry_run)
    prepare_dir(final_dir, force=args.force, dry_run=args.dry_run)
    run(
        [
            args.python,
            "scripts/transform_nemotron_parakeet_export.py",
            export_dir,
            "--quantize",
            "--projected-cache",
            *transform_extra_args,
            "--keep-fp32",
        ],
        dry_run=args.dry_run,
    )
    run(
        [
            args.python,
            "scripts/build_nemotron_fixed_shape_model.py",
            "--source-dir",
            export_dir,
            "--output-dir",
            fixed_dir,
            "--ort-optimize-final",
            args.ort_optimize_final,
            "--ort-optimize-threads",
            str(args.ort_optimize_threads),
        ],
        dry_run=args.dry_run,
    )
    run(
        [
            args.python,
            "scripts/dequantize_nemotron_matmul_blocks.py",
            "--source-dir",
            fixed_dir,
            "--output-dir",
            final_dir,
            "--include",
            "/feed_forward",
            "--ort-optimize-final",
            args.ort_optimize_final,
            "--ort-optimize-threads",
            str(args.ort_optimize_threads),
        ],
        dry_run=args.dry_run,
    )
    benchmark(
        args,
        label=benchmark_label,
        model_dir=final_dir,
        output_json=args.bench_root / output_json_name,
    )
    return final_dir


def build_fp32_current_projection(args: argparse.Namespace) -> Path:
    return build_fp32_transform_variant(
        args,
        stem="fp32-current-projection",
        transform_extra_args=[
            "--projected-cache-current-projection",
            "fp32",
        ],
        benchmark_label="fp32_current_projection",
        output_json_name="sayboard-fp32-current-projection-001.json",
    )


def build_per_channel_quantization(args: argparse.Namespace) -> Path:
    return build_fp32_transform_variant(
        args,
        stem="per-channel-quantization",
        transform_extra_args=[
            "--quantize-per-channel",
        ],
        benchmark_label="per_channel_quantization",
        output_json_name="sayboard-per-channel-quantization-001.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        choices=EXPERIMENTS,
        help="Experiment to run. Omit to run all pending harvest experiments.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--force", action="store_true", help="Overwrite experiment output directories.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Only build model variants.")
    parser.add_argument(
        "--skip-worker-build",
        action="store_true",
        help="Do not rebuild the release parakeet worker before benchmark runs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("build/model-variants/sayboard-harvest"),
    )
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=Path("build/parakeet-variant-bench"),
    )
    parser.add_argument(
        "--projected-int8-model-dir",
        type=Path,
        default=Path("models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56"),
        help="Existing projected-cache int8 model used as fixed-length experiment input.",
    )
    parser.add_argument(
        "--fp32-export-dir",
        type=Path,
        default=Path("build/model-variants/nemotron-fp32-projected"),
        help="Directory containing FP32 consolidated export artifacts.",
    )
    parser.add_argument(
        "--baseline-model-dir",
        type=Path,
        default=Path("build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort"),
    )
    parser.add_argument("--ort-optimize-final", choices=("disable", "basic", "extended", "all"), default="extended")
    parser.add_argument("--ort-optimize-threads", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--min-mem-available-gb", type=float, default=6.0)
    parser.add_argument("--child-memory-limit-gb", type=float, default=10.0)
    parser.add_argument("--power-profile", default="balanced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_worker(args)
    experiments = args.experiment or list(EXPERIMENTS)
    if "fixed-length" in experiments:
        build_fixed_length(args)
    if "fp32-current-projection" in experiments:
        build_fp32_current_projection(args)
    if "per-channel-quantization" in experiments:
        build_per_channel_quantization(args)


if __name__ == "__main__":
    main()
