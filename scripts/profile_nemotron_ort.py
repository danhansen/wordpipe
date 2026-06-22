#!/usr/bin/env python3
"""Profile one Nemotron encoder ONNX session with fixed streaming inputs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


ORT_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


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


def encoder_inputs(model_path: Path, input_frames: int, output_frames: int) -> dict[str, np.ndarray]:
    model = onnx.load(model_path, load_external_data=False)
    inputs: dict[str, np.ndarray] = {}
    for value_info in model.graph.input:
        name = value_info.name
        shape = tensor_shape(value_info)
        if name == "processed_signal":
            inputs[name] = np.zeros((1, 128, input_frames), dtype=np.float32)
        elif name == "processed_signal_length":
            inputs[name] = np.asarray([input_frames], dtype=np.int64)
        elif name == "prompt_index":
            inputs[name] = np.asarray([0], dtype=np.int64)
        elif name == "cache_last_channel":
            inputs[name] = np.zeros(tuple(int(dim) for dim in shape), dtype=np.float32)
        elif name == "cache_last_time":
            inputs[name] = np.zeros(tuple(int(dim) for dim in shape), dtype=np.float32)
        elif name == "cache_last_channel_len":
            inputs[name] = np.asarray([0], dtype=np.int64)
        elif name.startswith("cache_key_layer_") or name.startswith("cache_value_layer_"):
            inputs[name] = np.zeros(tuple(int(dim) for dim in shape), dtype=np.float32)
        else:
            raise SystemExit(f"Do not know how to synthesize encoder input {name!r} shape={shape}")

    # Keep output_frames in the signature for command provenance. The ONNX graph
    # determines the actual output count from input_frames and model internals.
    _ = output_frames
    return inputs


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
        "profile": str(path),
        "node_events": node_events,
        "total_node_time_ms": total_us / 1000,
        "time_by_provider_ms": [
            [name, duration / 1000]
            for name, duration in sorted(by_provider.items(), key=lambda item: item[1], reverse=True)
        ],
        "time_by_op_ms": [
            [name, duration / 1000]
            for name, duration in sorted(by_op.items(), key=lambda item: item[1], reverse=True)[:limit]
        ],
        "top_nodes_ms": [
            [name, duration / 1000]
            for name, duration in by_node.most_common(limit)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-frames", type=int, default=65)
    parser.add_argument("--output-frames", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--graph-optimization", choices=sorted(ORT_LEVELS), default="extended")
    parser.add_argument("--output-dir", type=Path, default=Path("build/ort-profile"))
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_dir / "encoder.onnx"
    if not model_path.exists():
        raise SystemExit(f"Missing encoder: {model_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    options = ort.SessionOptions()
    options.graph_optimization_level = ORT_LEVELS[args.graph_optimization]
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(args.output_dir / f"encoder-{args.graph_optimization}")

    inputs = encoder_inputs(model_path, args.input_frames, args.output_frames)
    started_load = time.perf_counter()
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    load_seconds = time.perf_counter() - started_load

    for _ in range(args.warmup):
        session.run(None, inputs)

    run_times = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        session.run(None, inputs)
        run_times.append(time.perf_counter() - started)

    profile_path = Path(session.end_profiling())
    profile = summarize_profile(profile_path, args.limit)
    result = {
        "model": str(model_path),
        "graph_optimization": args.graph_optimization,
        "threads": args.threads,
        "input_frames": args.input_frames,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_seconds": load_seconds,
        "mean_run_ms": sum(run_times) * 1000 / len(run_times),
        "min_run_ms": min(run_times) * 1000,
        "max_run_ms": max(run_times) * 1000,
        "profile": profile,
    }
    summary_path = args.output_dir / f"encoder-{args.graph_optimization}-summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
