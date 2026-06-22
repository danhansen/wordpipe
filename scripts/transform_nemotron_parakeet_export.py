#!/usr/bin/env python3
"""Transform a FP32 Nemotron parakeet export into the runtime int8 package.

Run this after export_nemotron_parakeet_optimized.py --export-only. Keeping
this in a separate process avoids holding the full NeMo/Torch model in memory
while ONNX Runtime quantizes the multi-GB encoder graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from rewrite_nemotron_projected_kv_cache import rewrite_model as rewrite_projected_cache

ORT_OPTIMIZATION_LEVELS = {"disable", "basic", "extended", "all"}


KEEP_FILES = {
    "encoder.fp32.consolidated.onnx",
    "encoder.fp32.consolidated.data",
    "decoder_joint.fp32.onnx",
    "tokenizer.model",
    "export_config.json",
    "config.json",
    "encoder.quant.onnx",
    "encoder.onnx",
    "decoder_joint.onnx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--projected-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply projected K/V cache rewrite to the quantized encoder.",
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run dynamic QUInt8 quantization.",
    )
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Keep FP32 encoder/decoder artifacts after transform.",
    )
    parser.add_argument(
        "--ort-optimize-final",
        choices=sorted(ORT_OPTIMIZATION_LEVELS),
        help="Serialize the final encoder through ONNX Runtime at the selected optimization level.",
    )
    parser.add_argument(
        "--ort-optimize-threads",
        type=int,
        default=1,
        help="Intra-op threads to use while serializing the ORT-optimized final encoder.",
    )
    return parser.parse_args()


def cleanup_export_shards(model_dir: Path) -> None:
    for path in model_dir.iterdir():
        if path.is_file() and path.name not in KEEP_FILES:
            path.unlink()


def quantize_to_single_file(input_path: Path, output_path: Path) -> None:
    print(f"[transform] quantizing {input_path.name} -> {output_path.name}", flush=True)
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        use_external_data_format=False,
    )
    onnx.checker.check_model(str(output_path))


def ort_optimize_to_file(input_path: Path, output_path: Path, level: str, threads: int) -> None:
    import onnxruntime as ort

    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    print(f"[transform] ORT optimizing {input_path.name} -> {output_path.name} level={level}", flush=True)
    options = ort.SessionOptions()
    options.optimized_model_filepath = str(output_path)
    options.graph_optimization_level = levels[level]
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    ort.InferenceSession(str(input_path), sess_options=options, providers=["CPUExecutionProvider"])
    onnx.checker.check_model(str(output_path))


def load_config(model_dir: Path) -> dict:
    for name in ("export_config.json", "config.json"):
        path = model_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    if not model_dir.is_dir():
        raise SystemExit(f"Missing model directory: {model_dir}")

    cleanup_export_shards(model_dir)

    encoder_fp32 = model_dir / "encoder.fp32.consolidated.onnx"
    decoder_fp32 = model_dir / "decoder_joint.fp32.onnx"
    if not encoder_fp32.exists() or not decoder_fp32.exists():
        raise SystemExit("Expected encoder.fp32.consolidated.onnx and decoder_joint.fp32.onnx")

    encoder_for_projected = encoder_fp32
    if args.quantize:
        encoder_quant = model_dir / "encoder.quant.onnx"
        decoder_quant = model_dir / "decoder_joint.onnx"
        quantize_to_single_file(encoder_fp32, encoder_quant)
        quantize_to_single_file(decoder_fp32, decoder_quant)
        encoder_for_projected = encoder_quant
    else:
        (model_dir / "decoder_joint.onnx").write_bytes(decoder_fp32.read_bytes())

    final_encoder = model_dir / "encoder.onnx"
    if args.projected_cache:
        print("[transform] rewriting encoder with projected cache", flush=True)
        rewrite_projected_cache(encoder_for_projected, final_encoder, "dynamic-int8")
    else:
        final_encoder.write_bytes(encoder_for_projected.read_bytes())

    if args.ort_optimize_final:
        optimized_encoder = model_dir / "encoder.ort_optimized.onnx"
        ort_optimize_to_file(
            final_encoder,
            optimized_encoder,
            args.ort_optimize_final,
            args.ort_optimize_threads,
        )
        optimized_encoder.replace(final_encoder)

    config = load_config(model_dir)
    config["projected_cache"] = args.projected_cache
    config["dynamic_quint8_quantization"] = args.quantize
    config["ort_optimized_final_encoder"] = args.ort_optimize_final
    (model_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if not args.keep_fp32:
        for name in (
            "encoder.fp32.consolidated.onnx",
            "encoder.fp32.consolidated.data",
            "encoder.quant.onnx",
            "decoder_joint.fp32.onnx",
            "export_config.json",
        ):
            (model_dir / name).unlink(missing_ok=True)

    print(f"[transform] wrote {model_dir}", flush=True)
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size / 1024 / 1024:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
