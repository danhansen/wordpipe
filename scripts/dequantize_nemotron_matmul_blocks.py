#!/usr/bin/env python3
"""Rewrite selected dynamic-quantized MatMul blocks back to float MatMul/Gemm."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

ORT_OPTIMIZATION_LEVELS = {"disable", "basic", "extended", "all"}


@dataclass(frozen=True)
class RewriteSpec:
    first_node_index: int
    nodes_to_remove: set[int]
    replacement_nodes: list[onnx.NodeProto]


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def tensor_shapes(model: onnx.ModelProto) -> dict[str, list[int | str | None]]:
    shapes: dict[str, list[int | str | None]] = {}
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int | str | None] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        shapes[value_info.name] = dims
    return shapes


def static_shape(shape: list[int | str | None] | None) -> list[int] | None:
    if shape is None:
        return None
    out = []
    for dim in shape:
        if not isinstance(dim, int):
            return None
        out.append(dim)
    return out


def const_array(initializers: dict[str, onnx.TensorProto], name: str) -> np.ndarray:
    return numpy_helper.to_array(initializers[name])


def only_consumer(consumers: dict[str, list[int]], tensor_name: str, expected_op: str, nodes: list[onnx.NodeProto]) -> int | None:
    use_sites = consumers.get(tensor_name, [])
    if len(use_sites) != 1:
        return None
    index = use_sites[0]
    return index if nodes[index].op_type == expected_op else None


def matches_name(name: str, include: list[str], exclude: list[str]) -> bool:
    if include and not any(pattern in name for pattern in include):
        return False
    if exclude and any(pattern in name for pattern in exclude):
        return False
    return True


def build_rewrite_specs(model: onnx.ModelProto, include: list[str], exclude: list[str]) -> list[RewriteSpec]:
    try:
        shape_model = shape_inference.infer_shapes(model)
    except Exception:
        shape_model = model

    initializers = {init.name: init for init in model.graph.initializer}
    shapes = tensor_shapes(shape_model)
    nodes = list(model.graph.node)
    producers: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for output in node.output:
            producers[output] = index
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(index)

    specs: list[RewriteSpec] = []
    for matmul_index, node in enumerate(nodes):
        if node.op_type != "MatMulInteger" or len(node.input) != 4 or len(node.output) != 1:
            continue
        if not matches_name(node.name or node.output[0], include, exclude):
            continue

        lhs_quant, rhs_quant, lhs_zero_point, rhs_zero_point = node.input
        if rhs_quant not in initializers or rhs_zero_point not in initializers:
            continue

        dql_index = producers.get(lhs_quant)
        if dql_index is None:
            continue
        dql = nodes[dql_index]
        if dql.op_type != "DynamicQuantizeLinear" or len(dql.output) != 3:
            continue
        if dql.output[0] != lhs_quant or dql.output[2] != lhs_zero_point:
            continue
        lhs_input = dql.input[0]
        lhs_scale = dql.output[1]

        cast_index = only_consumer(consumers, node.output[0], "Cast", nodes)
        if cast_index is None:
            continue
        cast_node = nodes[cast_index]

        scale_mul_index = None
        rhs_scale_name = None
        for candidate_index in consumers.get(lhs_scale, []):
            candidate = nodes[candidate_index]
            if candidate.op_type != "Mul" or len(candidate.input) != 2:
                continue
            other_input = candidate.input[0] if candidate.input[1] == lhs_scale else candidate.input[1]
            if other_input in initializers:
                scale_mul_index = candidate_index
                rhs_scale_name = other_input
                break
        if scale_mul_index is None or rhs_scale_name is None:
            continue
        scale_mul_output = nodes[scale_mul_index].output[0]

        output_mul_index = None
        for candidate_index in consumers.get(cast_node.output[0], []):
            candidate = nodes[candidate_index]
            if candidate.op_type == "Mul" and scale_mul_output in candidate.input:
                output_mul_index = candidate_index
                break
        if output_mul_index is None:
            continue

        matmul_float_output = nodes[output_mul_index].output[0]
        final_output_name = matmul_float_output
        bias_add_index = None
        bias_name = None
        for candidate_index in consumers.get(final_output_name, []):
            candidate = nodes[candidate_index]
            if candidate.op_type != "Add" or len(candidate.input) != 2:
                continue
            other_input = candidate.input[0] if candidate.input[1] == final_output_name else candidate.input[1]
            if other_input in initializers:
                bias_add_index = candidate_index
                bias_name = other_input
                final_output_name = candidate.output[0]
                break

        lhs_input_shape = static_shape(shapes.get(lhs_input))
        final_output_shape = static_shape(shapes.get(final_output_name))

        quantized = const_array(initializers, rhs_quant).astype(np.float32)
        scale = const_array(initializers, rhs_scale_name).astype(np.float32)
        zero_point = const_array(initializers, rhs_zero_point).astype(np.float32)
        if scale.size != 1 or zero_point.size != 1 or quantized.ndim != 2:
            continue
        weight = (quantized - float(zero_point.reshape(()))) * float(scale.reshape(()))

        base = (node.name or node.output[0]).replace("/", "_")
        weight_name = f"{base}_dequant_weight"
        model.graph.initializer.append(numpy_helper.from_array(weight.astype(np.float32), weight_name))

        replacement_nodes: list[onnx.NodeProto] = []
        gemm_input = lhs_input
        gemm_output = final_output_name
        use_gemm = bias_name is not None
        needs_output_reshape = False
        if lhs_input_shape is not None and final_output_shape is not None and len(lhs_input_shape) > 2:
            leading = int(np.prod(lhs_input_shape[:-1], dtype=np.int64))
            inner = lhs_input_shape[-1]
            reshape_in_shape = f"{base}_reshape_in_shape"
            reshape_out_shape = f"{base}_reshape_out_shape"
            reshape_in_output = f"{base}_reshape_in_output"
            gemm_output = f"{base}_matmul_output"
            model.graph.initializer.extend(
                [
                    numpy_helper.from_array(np.asarray([leading, inner], dtype=np.int64), reshape_in_shape),
                    numpy_helper.from_array(np.asarray(final_output_shape, dtype=np.int64), reshape_out_shape),
                ]
            )
            replacement_nodes.append(
                helper.make_node("Reshape", [lhs_input, reshape_in_shape], [reshape_in_output], name=f"{base}_reshape_in")
            )
            gemm_input = reshape_in_output
            needs_output_reshape = True
        elif use_gemm and (lhs_input_shape is None or len(lhs_input_shape) != 2):
            use_gemm = False
            gemm_output = matmul_float_output

        if use_gemm:
            inputs = [gemm_input, weight_name, bias_name]
            replacement_nodes.append(
                helper.make_node(
                    "Gemm",
                    inputs,
                    [gemm_output],
                    name=f"{base}_gemm_dequantized",
                    alpha=1.0,
                    beta=1.0,
                    transA=0,
                    transB=0,
                )
            )
        else:
            replacement_nodes.append(
                helper.make_node(
                    "MatMul",
                    [gemm_input, weight_name],
                    [gemm_output],
                    name=f"{base}_matmul_dequantized",
                )
            )

        if needs_output_reshape:
            reshape_out_shape = f"{base}_reshape_out_shape"
            replacement_nodes.append(
                helper.make_node("Reshape", [gemm_output, reshape_out_shape], [final_output_name], name=f"{base}_reshape_out")
            )

        # Some projections, notably self_attn/linear_pos, share one
        # DynamicQuantizeLinear output across many MatMulInteger consumers.
        # Keep the DQL and scale-mul nodes in place; ORT can prune dead tails
        # after all selected matmuls have been replaced.
        _ = dql_index
        _ = scale_mul_index
        remove = {matmul_index, cast_index, output_mul_index}
        if bias_add_index is not None and use_gemm:
            remove.add(bias_add_index)
        specs.append(RewriteSpec(min(remove), remove, replacement_nodes))
    return specs


def apply_rewrites(model: onnx.ModelProto, specs: list[RewriteSpec]) -> int:
    by_first = {spec.first_node_index: spec for spec in specs}
    removed = {index for spec in specs for index in spec.nodes_to_remove}
    rewritten: list[onnx.NodeProto] = []
    for index, node in enumerate(model.graph.node):
        spec = by_first.get(index)
        if spec is not None:
            rewritten.extend(spec.replacement_nodes)
        if index in removed:
            continue
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return len(specs)


def prune_unused_initializers(model: onnx.ModelProto) -> int:
    used = {input_name for node in model.graph.node for input_name in node.input if input_name}
    kept = [initializer for initializer in model.graph.initializer if initializer.name in used]
    removed = len(model.graph.initializer) - len(kept)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    return removed


def ort_optimize_to_file(input_path: Path, output_path: Path, level: str, threads: int) -> None:
    import onnxruntime as ort

    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    options = ort.SessionOptions()
    options.optimized_model_filepath = str(output_path)
    options.graph_optimization_level = levels[level]
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    ort.InferenceSession(str(input_path), sess_options=options, providers=["CPUExecutionProvider"])
    onnx.checker.check_model(str(output_path))


def update_config(
    source_dir: Path,
    output_dir: Path,
    *,
    include: list[str],
    exclude: list[str],
    rewritten: int,
    pruned_initializers: int,
    ort_optimize_final: str | None,
) -> None:
    source_config = source_dir / "config.json"
    if not source_config.exists():
        return
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["dequantized_matmul_blocks"] = {
        "include": include,
        "exclude": exclude,
        "rewritten_blocks": rewritten,
        "pruned_initializers": pruned_initializers,
        "ort_optimized_final_encoder": ort_optimize_final,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def build_model_dir(
    source_dir: Path,
    output_dir: Path,
    include: list[str],
    exclude: list[str],
    ort_optimize_final: str | None,
    ort_optimize_threads: int,
) -> None:
    source_encoder = source_dir / "encoder.onnx"
    if not source_encoder.exists():
        raise SystemExit(f"Missing source encoder: {source_encoder}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(source_encoder, load_external_data=False)
    specs = build_rewrite_specs(model, include, exclude)
    rewritten = apply_rewrites(model, specs)
    pruned_initializers = prune_unused_initializers(model)
    output_encoder = output_dir / "encoder.onnx"
    onnx.save(model, output_encoder)
    onnx.checker.check_model(str(output_encoder))
    if ort_optimize_final:
        optimized_encoder = output_dir / "encoder.ort_optimized.onnx"
        ort_optimize_to_file(output_encoder, optimized_encoder, ort_optimize_final, ort_optimize_threads)
        optimized_encoder.replace(output_encoder)

    for sidecar in ("decoder_joint.onnx", "tokenizer.model"):
        src = source_dir / sidecar
        if src.exists():
            hardlink_or_copy(src, output_dir / sidecar)
    update_config(
        source_dir,
        output_dir,
        include=include,
        exclude=exclude,
        rewritten=rewritten,
        pruned_initializers=pruned_initializers,
        ort_optimize_final=ort_optimize_final,
    )

    print(
        f"[dequantize] wrote {output_dir} rewritten_blocks={rewritten} "
        f"prunedInitializers={pruned_initializers} ortFinal={ort_optimize_final}"
    )
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size / 1024 / 1024:.1f} MiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include", action="append", default=[], help="Substring that matched MatMulInteger node names must contain.")
    parser.add_argument("--exclude", action="append", default=[], help="Substring that matched MatMulInteger node names must not contain.")
    parser.add_argument("--ort-optimize-final", choices=sorted(ORT_OPTIMIZATION_LEVELS))
    parser.add_argument("--ort-optimize-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_model_dir(
        args.source_dir,
        args.output_dir,
        args.include,
        args.exclude,
        args.ort_optimize_final,
        args.ort_optimize_threads,
    )


if __name__ == "__main__":
    main()
