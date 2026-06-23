#!/usr/bin/env python3
"""Run a single Microsoft Olive ONNX pass on a Wordpipe model directory.

This is an experiment wrapper, not the main export pipeline. It keeps the
Wordpipe model layout intact while making Olive runs reproducible:

  source/
    encoder.onnx
    decoder_joint.onnx
    config.json
    tokenizer.model

  output/
    encoder.onnx
    decoder_joint.onnx
    config.json
    tokenizer.model
    olive_pass_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


GRAPH_FILES = {
    "encoder": "encoder.onnx",
    "decoder_joint": "decoder_joint.onnx",
}
SUPPORT_FILES = ("config.json", "tokenizer.model")


def parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    path = Path(raw)
    text = path.read_text(encoding="utf-8") if path.exists() else raw
    value = json.loads(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("pass config must be a JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--pass-name",
        choices=("peephole", "quant-preprocess", "dynamic-quant", "ort-transformers"),
        default="peephole",
    )
    parser.add_argument(
        "--pass-config",
        type=parse_json_object,
        default=None,
        help="Inline JSON object or path to a JSON file with Olive pass config.",
    )
    parser.add_argument(
        "--component",
        choices=tuple(GRAPH_FILES),
        action="append",
        help="Component to transform. Defaults to both encoder and decoder_joint.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite the output directory.")
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


def copy_support_files(source_dir: Path, output_dir: Path) -> None:
    for name in SUPPORT_FILES:
        source_path = source_dir / name
        require_file(source_path)
        shutil.copy2(source_path, output_dir / name)


def copy_graph_with_external_data(source_graph: Path, output_graph: Path) -> None:
    shutil.copy2(source_graph, output_graph)
    external = source_graph.with_name(f"{source_graph.name}.data")
    if external.exists():
        shutil.copy2(external, output_graph.with_name(external.name))


def pass_class(pass_name: str):
    if pass_name == "peephole":
        from olive.passes.onnx.peephole_optimizer import OnnxPeepholeOptimizer

        return OnnxPeepholeOptimizer
    if pass_name == "quant-preprocess":
        from olive.passes.onnx.quantization import OnnxQuantizationPreprocess

        return OnnxQuantizationPreprocess
    if pass_name == "dynamic-quant":
        from olive.passes.onnx.quantization import OnnxDynamicQuantization

        return OnnxDynamicQuantization
    if pass_name == "ort-transformers":
        from olive.passes.onnx.transformer_optimization import OrtTransformersOptimization

        return OrtTransformersOptimization
    raise ValueError(f"unknown pass: {pass_name}")


def graph_summary(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    ops = Counter(node.op_type for node in model.graph.node)
    external_data_size = sum(
        sibling.stat().st_size
        for sibling in sorted(path.parent.glob(f"{path.name}.data*"))
        if sibling.is_file()
    )
    return {
        "path": str(path),
        "onnx_size_bytes": path.stat().st_size,
        "external_data_size_bytes": external_data_size,
        "total_size_bytes": path.stat().st_size + external_data_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "ops": dict(sorted(ops.items())),
        "top_ops": ops.most_common(20),
    }


def delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_ops = before["ops"]
    after_ops = after["ops"]
    changed_ops = {}
    for op in sorted(set(before_ops) | set(after_ops)):
        old = before_ops.get(op, 0)
        new = after_ops.get(op, 0)
        if old != new:
            changed_ops[op] = {"before": old, "after": new, "delta": new - old}
    return {
        "nodes": after["nodes"] - before["nodes"],
        "initializers": after["initializers"] - before["initializers"],
        "total_size_bytes": after["total_size_bytes"] - before["total_size_bytes"],
        "changed_ops": changed_ops,
    }


def default_pass_config(pass_name: str) -> dict[str, Any]:
    if pass_name == "peephole":
        return {
            "onnxscript_optimize": True,
            "onnxoptimizer_optimize": True,
            "fuse_reshape_operations": True,
            "save_as_external_data": False,
        }
    if pass_name == "quant-preprocess":
        return {
            "skip_optimization": False,
            "skip_onnx_shape": False,
            "skip_symbolic_shape": False,
            "save_as_external_data": False,
        }
    if pass_name == "dynamic-quant":
        return {
            "precision": "int8",
            "per_channel": False,
            "reduce_range": False,
            "quant_preprocess": True,
            "save_as_external_data": False,
        }
    if pass_name == "ort-transformers":
        return {
            "opt_level": 99,
            "only_onnxruntime": True,
            "use_gpu": False,
            "save_as_external_data": False,
        }
    return {}


def main() -> None:
    args = parse_args()
    # Olive imports matplotlib through some optional paths. Keep its cache in
    # the repo build tree to avoid per-run /tmp cache churn.
    os.environ.setdefault("MPLCONFIGDIR", str(Path("build/matplotlib-cache").resolve()))

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Source is not a directory: {source_dir}")

    prepare_output(output_dir, force=args.force)
    copy_support_files(source_dir, output_dir)

    from olive.hardware import DEFAULT_CPU_ACCELERATOR
    from olive.model import ONNXModelHandler
    from olive.passes.olive_pass import create_pass_from_dict

    config = default_pass_config(args.pass_name)
    config.update(args.pass_config or {})
    selected_components = args.component or list(GRAPH_FILES)
    cls = pass_class(args.pass_name)
    olive_pass = create_pass_from_dict(
        cls,
        config,
        disable_search=True,
        accelerator_spec=DEFAULT_CPU_ACCELERATOR,
    )

    summary: dict[str, Any] = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "pass_name": args.pass_name,
        "pass_config": config,
        "components": {},
    }

    for component in selected_components:
        graph_name = GRAPH_FILES[component]
        source_graph = source_dir / graph_name
        output_graph = output_dir / graph_name
        require_file(source_graph)
        before = graph_summary(source_graph)
        print(f"[olive] {args.pass_name}: {component} -> {output_graph}", flush=True)
        output_model = olive_pass.run(ONNXModelHandler(model_path=str(source_graph)), str(output_graph))
        returned_graph = Path(output_model.model_path)
        no_op_return = returned_graph.resolve() == source_graph.resolve()
        if not output_graph.exists():
            if not returned_graph.exists():
                raise SystemExit(f"Olive did not write an output model: {output_graph}")
            copy_graph_with_external_data(returned_graph, output_graph)
        after = graph_summary(output_graph)
        component_summary = {
            "before": before,
            "after": after,
            "delta": delta(before, after),
            "no_op_return": no_op_return,
            "returned_model_path": str(returned_graph),
        }
        summary["components"][component] = component_summary
        print(
            "[olive] "
            f"{component}: nodes {before['nodes']} -> {after['nodes']} "
            f"({component_summary['delta']['nodes']:+d}), "
            f"size {before['total_size_bytes']} -> {after['total_size_bytes']} bytes",
            flush=True,
        )

    # Components not transformed should still be present in the output model dir.
    for component, graph_name in GRAPH_FILES.items():
        if component in selected_components:
            continue
        source_graph = source_dir / graph_name
        require_file(source_graph)
        copy_graph_with_external_data(source_graph, output_dir / graph_name)

    summary_path = output_dir / "olive_pass_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[olive] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
