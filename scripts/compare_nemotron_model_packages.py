#!/usr/bin/env python3
"""Compare two Wordpipe/Nemotron model package directories."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import onnx


def file_size_mib(path: Path) -> float | None:
    return path.stat().st_size / 1024 / 1024 if path.exists() else None


def value_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
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


def value_summary(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "elem_type": int(tensor_type.elem_type),
        "shape": value_shape(value_info),
    }


def graph_summary(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    op_counts = Counter(node.op_type for node in model.graph.node)
    domain_counts = Counter(node.domain or "" for node in model.graph.node)
    initializer_data_locations = Counter(int(init.data_location) for init in model.graph.initializer)
    return {
        "ir_version": int(model.ir_version),
        "opsets": {opset.domain or "": int(opset.version) for opset in model.opset_import},
        "inputs": [value_summary(value) for value in model.graph.input],
        "outputs": [value_summary(value) for value in model.graph.output],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "value_info_count": len(model.graph.value_info),
        "op_counts": dict(sorted(op_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "initializer_data_locations": dict(sorted(initializer_data_locations.items())),
        "metadata": {item.key: item.value for item in model.metadata_props},
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def package_summary(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "files_mib": {
            name: file_size_mib(path / name)
            for name in (
                "encoder.onnx",
                "encoder.onnx.data",
                "decoder_joint.onnx",
                "tokenizer.model",
                "config.json",
            )
        },
        "config": load_json(path / "config.json"),
        "encoder": graph_summary(path / "encoder.onnx"),
        "decoder_joint": graph_summary(path / "decoder_joint.onnx"),
    }


def diff_values(left: Any, right: Any) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        out = {}
        for key in keys:
            if key not in left:
                out[key] = {"left": None, "right": right[key]}
            elif key not in right:
                out[key] = {"left": left[key], "right": None}
            else:
                child = diff_values(left[key], right[key])
                if child:
                    out[key] = child
        return out
    if left != right:
        return {"left": left, "right": right}
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = package_summary(args.left)
    right = package_summary(args.right)
    report = {
        "left": left,
        "right": right,
        "diff": {
            "files_mib": diff_values(left["files_mib"], right["files_mib"]),
            "config": diff_values(left["config"], right["config"]),
            "encoder": diff_values(left["encoder"], right["encoder"]),
            "decoder_joint": diff_values(left["decoder_joint"], right["decoder_joint"]),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
