#!/usr/bin/env python3
"""Build a model directory with encoder MatMul weights quantized to MatMulNBits."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
from onnxruntime.quantization.neural_compressor.weight_only import rtn_quantize

logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
logging.getLogger("neural_compressor").setLevel(logging.WARNING)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def matches_name(name: str, include: list[str], exclude: list[str]) -> bool:
    if include and not any(pattern in name for pattern in include):
        return False
    if exclude and any(pattern in name for pattern in exclude):
        return False
    return True


def matmul_weight_config(
    model: onnx.ModelProto,
    include: list[str],
    exclude: list[str],
    bits: int,
    block_size: int,
) -> tuple[dict[str, object], list[str]]:
    initializers = {init.name for init in model.graph.initializer}
    config: dict[str, object] = {}
    selected: list[str] = []
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2 or node.input[1] not in initializers:
            continue
        name = node.name or node.output[0]
        if matches_name(name, include, exclude):
            config[node.name] = {"bits": bits, "group_size": block_size, "scheme": "asym"}
            selected.append(node.name)
        else:
            config[node.name] = "fp32"
    return config, selected


def quantize_encoder(args: argparse.Namespace) -> None:
    source_encoder = args.source_dir / "encoder.onnx"
    if not source_encoder.exists():
        raise SystemExit(f"Missing source encoder: {source_encoder}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(source_encoder)
    config, selected_names = matmul_weight_config(model, args.include, args.exclude, args.bits, args.block_size)
    selected = len(selected_names)
    if selected == 0:
        raise SystemExit("No static-RHS MatMul nodes matched the requested filters.")
    if args.dry_run:
        print(
            f"[matmul-nbits] dry-run source={args.source_dir} "
            f"selected_matmul_nodes={selected} bits={args.bits} block_size={args.block_size} algorithm={args.algorithm}"
        )
        return

    output_encoder = args.output_dir / "encoder.onnx"
    if args.algorithm == "default":
        quant_config = DefaultWeightOnlyQuantConfig(
            block_size=args.block_size,
            is_symmetric=False,
            bits=args.bits,
            op_types_to_quantize=("MatMul",),
        )
        quantized = MatMulNBitsQuantizer(
            model=model,
            bits=args.bits,
            block_size=args.block_size,
            nodes_to_include=selected_names,
            algo_config=quant_config,
        )
        quantized.process()
        quantized.model.save_model_to_file(output_encoder.as_posix(), use_external_data_format=True)
    else:
        quantized = rtn_quantize(
            model=model,
            weight_config=config,
            num_bits=args.bits,
            group_size=args.block_size,
            scheme="asym",
            algorithm=args.algorithm,
        )
        quantized.save(output_encoder.as_posix())
    onnx.checker.check_model(output_encoder.as_posix())

    for sidecar in ("decoder_joint.onnx", "tokenizer.model", "config.json"):
        src = args.source_dir / sidecar
        if src.exists():
            hardlink_or_copy(src, args.output_dir / sidecar)

    summary = {
        "source_dir": str(args.source_dir),
        "bits": args.bits,
        "block_size": args.block_size,
        "algorithm": args.algorithm,
        "include": args.include,
        "exclude": args.exclude,
        "selected_matmul_nodes": selected,
    }
    (args.output_dir / "matmul_nbits_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[matmul-nbits] wrote {args.output_dir} "
        f"selected_matmul_nodes={selected} bits={args.bits} block_size={args.block_size} algorithm={args.algorithm}"
    )
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size / 1024 / 1024:.1f} MiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--algorithm", choices=("default", "k_quant", "RTN"), default="k_quant")
    parser.add_argument("--dry-run", action="store_true", help="Only count selected MatMul nodes.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only quantize MatMul nodes whose name/output contains this substring. Repeatable.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Do not quantize MatMul nodes whose name/output contains this substring. Repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    quantize_encoder(parse_args())


if __name__ == "__main__":
    main()
