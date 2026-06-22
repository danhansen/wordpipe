#!/usr/bin/env python3
"""Summarize ONNX graphs and optional ONNX Runtime optimized/profile output."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx


ORT_LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


def mb(value: int) -> float:
    return value / 1024 / 1024


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    dims: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def tensor_bytes(tensor: onnx.TensorProto) -> int:
    try:
        return len(tensor.raw_data)
    except Exception:
        return 0


def model_summary(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    op_counts = Counter(node.op_type for node in model.graph.node)
    domain_counts = Counter(node.domain or "ai.onnx" for node in model.graph.node)
    dtype_bytes: Counter[str] = Counter()
    for initializer in model.graph.initializer:
        dtype = onnx.TensorProto.DataType.Name(initializer.data_type)
        dtype_bytes[dtype] += tensor_bytes(initializer)

    unresolved_dims = 0
    value_infos = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value_info in value_infos:
        for dim in tensor_shape(value_info):
            if not isinstance(dim, int):
                unresolved_dims += 1

    names = [node.name for node in model.graph.node]
    attention_projection_patterns = [
        "/self_attn/linear_q/",
        "/self_attn/linear_k/",
        "/self_attn/linear_v/",
        "/self_attn/linear_out/",
        "/self_attn/linear_pos/",
    ]
    categories = {
        "attention_projection": sum(
            1 for name in names if any(pattern in name for pattern in attention_projection_patterns)
        ),
        "attention_score_context": sum(1 for name in names if "/self_attn/MatMul" in name),
        "feed_forward": sum(1 for name in names if "/feed_forward" in name),
        "pre_encode_conv": sum(1 for name in names if "/pre_encode/conv/" in name),
        "projected_cache": sum(1 for name in names if "projected" in name or "cache_key_layer" in name),
    }

    return {
        "path": str(path),
        "size_mb": round(mb(path.stat().st_size), 3) if path.exists() else None,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "inputs": [
            {"name": value.name, "shape": tensor_shape(value)}
            for value in model.graph.input
        ],
        "outputs": [
            {"name": value.name, "shape": tensor_shape(value)}
            for value in model.graph.output
        ],
        "domains": dict(sorted(domain_counts.items())),
        "ops": dict(sorted(op_counts.items())),
        "top_ops": op_counts.most_common(20),
        "initializer_bytes_by_dtype": {
            dtype: round(mb(bytes_), 3) for dtype, bytes_ in sorted(dtype_bytes.items())
        },
        "unresolved_shape_dims": unresolved_dims,
        "name_categories": categories,
        "quantization_ops": {
            key: op_counts.get(key, 0)
            for key in [
                "DynamicQuantizeLinear",
                "MatMulInteger",
                "DynamicQuantizeMatMul",
                "MatMulIntegerToFloat",
                "QLinearMatMul",
                "QLinearConv",
                "ConvInteger",
            ]
        },
    }


def emit_optimized_model(input_path: Path, output_path: Path, level: str, threads: int) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.optimized_model_filepath = str(output_path)
    options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, ORT_LEVELS[level])
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ort.InferenceSession(str(input_path), sess_options=options, providers=["CPUExecutionProvider"])


def summarize_profile(path: Path, limit: int) -> dict[str, Any]:
    events = json.loads(path.read_text(encoding="utf-8"))
    by_op: dict[str, int] = defaultdict(int)
    by_provider: dict[str, int] = defaultdict(int)
    by_node: Counter[str] = Counter()
    node_events = 0
    total_us = 0

    for event in events:
        if event.get("ph") != "X":
            continue
        args = event.get("args") or {}
        op = args.get("op_name") or args.get("op") or args.get("name")
        provider = args.get("provider")
        duration = int(event.get("dur") or 0)
        if not op and not provider:
            continue
        node_events += 1
        total_us += duration
        if op:
            by_op[str(op)] += duration
        if provider:
            by_provider[str(provider)] += duration
        by_node[str(event.get("name", "<unnamed>"))] += duration

    return {
        "path": str(path),
        "node_events": node_events,
        "total_node_time_ms": round(total_us / 1000, 3),
        "time_by_op_ms": [
            [name, round(duration / 1000, 3)]
            for name, duration in sorted(by_op.items(), key=lambda item: item[1], reverse=True)[:limit]
        ],
        "time_by_provider_ms": [
            [name, round(duration / 1000, 3)]
            for name, duration in sorted(by_provider.items(), key=lambda item: item[1], reverse=True)
        ],
        "top_nodes_ms": [
            [name, round(duration / 1000, 3)]
            for name, duration in by_node.most_common(limit)
        ],
    }


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n== {title} ==")
    print(f"path: {summary['path']}")
    if "nodes" in summary:
        print(
            f"size_mb={summary['size_mb']} nodes={summary['nodes']} "
            f"initializers={summary['initializers']} unresolved_dims={summary['unresolved_shape_dims']}"
        )
        print(f"domains: {summary['domains']}")
        print(f"top_ops: {summary['top_ops']}")
        print(f"quantization_ops: {summary['quantization_ops']}")
        print(f"name_categories: {summary['name_categories']}")
        print(f"initializer_mb_by_dtype: {summary['initializer_bytes_by_dtype']}")
    else:
        print(f"node_events={summary['node_events']} total_node_time_ms={summary['total_node_time_ms']}")
        print(f"time_by_provider_ms: {summary['time_by_provider_ms']}")
        print(f"time_by_op_ms: {summary['time_by_op_ms']}")
        print(f"top_nodes_ms: {summary['top_nodes_ms']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="*", help="ONNX model(s) to summarize.")
    parser.add_argument(
        "--emit-optimized",
        action="store_true",
        help="Ask ORT to write optimized model(s), then summarize them too.",
    )
    parser.add_argument(
        "--opt-level",
        choices=sorted(ORT_LEVELS),
        action="append",
        default=None,
        help="ORT optimization level to emit. May be repeated. Default: all.",
    )
    parser.add_argument("--threads", type=int, default=1, help="ORT intra-op threads for optimization dumps.")
    parser.add_argument("--output-dir", type=Path, default=Path("build/ort-diagnostics"))
    parser.add_argument("--profile-json", type=Path, action="append", help="ORT profile JSON to summarize.")
    parser.add_argument("--limit", type=int, default=20, help="Top-N rows in printed profile summaries.")
    parser.add_argument("--json-out", type=Path, help="Optional machine-readable JSON summary output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[dict[str, Any]] = []
    opt_levels = args.opt_level or ["all"]

    for model_path in args.model:
        source = model_summary(model_path)
        print_summary(model_path.name, source)
        results.append({"kind": "model", "level": "source", **source})

        if not args.emit_optimized:
            continue
        for level in opt_levels:
            out = args.output_dir / model_path.stem / f"{model_path.stem}.ort_{level}.onnx"
            print(f"\n[ort] writing optimized model level={level}: {out}", flush=True)
            emit_optimized_model(model_path, out, level, args.threads)
            optimized = model_summary(out)
            print_summary(f"{model_path.name} ORT {level}", optimized)
            results.append({"kind": "model", "level": level, **optimized})

    for profile_path in args.profile_json or []:
        profile = summarize_profile(profile_path, args.limit)
        print_summary(f"profile {profile_path.name}", profile)
        results.append({"kind": "profile", **profile})

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n[json] wrote {args.json_out}")


if __name__ == "__main__":
    main()
