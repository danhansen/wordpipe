#!/usr/bin/env python3
"""Rewrite Nemotron streaming encoder ONNX to cache projected K/V tensors.

The stock streaming encoder caches raw per-layer channel activations and then
reprojects the full 70-frame cache through linear_k/linear_v on every chunk.
This rewrite changes those attention K/V projections to:

    concat(caller_projected_cache, project(current_chunk))

The caller is then responsible for rolling the projected K/V cache using the
new projected_current_* graph outputs.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


@dataclass(frozen=True)
class ProjectedCacheSpec:
    num_layers: int
    batch_size: int
    cache_len: int
    hidden_size: int


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


def graph_value_infos(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    return {
        value_info.name: value_info
        for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    }


def static_dim(shape: list[int | str | None], index: int, name: str) -> int:
    if index >= len(shape) or not isinstance(shape[index], int):
        raise SystemExit(f"{name} needs static dimension {index}; got {shape}")
    return shape[index]


def infer_spec(model: onnx.ModelProto) -> ProjectedCacheSpec:
    values = graph_value_infos(model)
    cache_channel = values.get("cache_last_channel")
    if cache_channel is None:
        raise SystemExit("Expected cache_last_channel graph input.")
    cache_shape = tensor_shape(cache_channel)
    return ProjectedCacheSpec(
        num_layers=static_dim(cache_shape, 0, "cache_last_channel"),
        batch_size=static_dim(cache_shape, 1, "cache_last_channel"),
        cache_len=static_dim(cache_shape, 2, "cache_last_channel"),
        hidden_size=static_dim(cache_shape, 3, "cache_last_channel"),
    )


def add_initializer_once(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    if any(init.name == name for init in model.graph.initializer):
        return
    model.graph.initializer.append(numpy_helper.from_array(value, name))


def add_graph_input_once(
    model: onnx.ModelProto,
    name: str,
    elem_type: int,
    shape: list[int | str],
) -> None:
    if any(value.name == name for value in model.graph.input):
        return
    model.graph.input.append(helper.make_tensor_value_info(name, elem_type, shape))


def add_graph_output_once(
    model: onnx.ModelProto,
    name: str,
    elem_type: int,
    shape: list[int | str],
) -> None:
    if any(value.name == name for value in model.graph.output):
        return
    model.graph.output.append(helper.make_tensor_value_info(name, elem_type, shape))


def find_node(model: onnx.ModelProto, name: str) -> onnx.NodeProto:
    for node in model.graph.node:
        if node.name == name:
            return node
    raise SystemExit(f"Missing expected node: {name}")


def producer_by_output(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output}


def consumers_by_input(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
    return consumers


def layer_prefix(layer: int) -> str:
    return f"/encoder/layers.{layer}/self_attn"


def cache_layer_name(kind: str, layer: int) -> str:
    return f"cache_{kind}_layer_{layer}"


def projected_current_name(kind: str, layer: int) -> str:
    return f"projected_current_{kind}_layer_{layer}"


def projection_node_name(layer: int, kind: str) -> str:
    return f"{layer_prefix(layer)}/linear_{kind}/MatMul_quant"


def layer_from_projection_node(node_name: str) -> int:
    match = re.search(r"/encoder/layers\.(\d+)/self_attn/linear_[kv]/MatMul_quant$", node_name)
    if not match:
        raise SystemExit(f"Cannot infer layer from projection node name: {node_name}")
    return int(match.group(1))


def find_current_source(model: onnx.ModelProto, layer: int, projection_input: str) -> str:
    producers = producer_by_output(model)
    quantize = producers.get(projection_input)
    if quantize is None or quantize.op_type != "DynamicQuantizeLinear":
        raise SystemExit(f"Layer {layer} K/V projection input is not DynamicQuantizeLinear output.")
    concat = producers.get(quantize.input[0])
    if concat is None or concat.op_type != "Concat" or len(concat.input) < 2:
        raise SystemExit(f"Layer {layer} K/V source is not the expected raw-cache/current Concat.")
    return concat.input[1]


def projection_replacement(model: onnx.ModelProto, node: onnx.NodeProto) -> tuple[str, str, str, str]:
    """Return quantized weight, weight scale, weight zero point, final float output."""
    if node.op_type != "MatMulInteger":
        raise SystemExit(f"Unsupported K/V projection node type for {node.name}: {node.op_type}")

    scale_node = find_node(model, f"{node.name}_scales_mul")
    output_scale_node = find_node(model, f"{node.name}_output_scale_mul")
    if len(node.input) < 4 or len(scale_node.input) < 2 or not output_scale_node.output:
        raise SystemExit(f"Unexpected MatMulInteger quantization tail for {node.name}")
    return node.input[1], scale_node.input[1], node.input[3], output_scale_node.output[0]


def obsolete_projection_tail_names(model: onnx.ModelProto, node: onnx.NodeProto) -> set[str]:
    if node.op_type != "MatMulInteger":
        raise SystemExit(f"Unsupported K/V projection node type for {node.name}: {node.op_type}")

    remove = {node.name, f"{node.name}_scales_mul", f"{node.name}_output_scale_mul"}
    consumers = consumers_by_input(model)
    for consumer in consumers.get(node.output[0], []):
        if consumer.op_type != "Cast":
            raise SystemExit(f"Unexpected consumer of {node.name}: {consumer.name} {consumer.op_type}")
        remove.add(consumer.name)
    return remove


def dequantized_weight_initializer(model: onnx.ModelProto, node: onnx.NodeProto, name: str) -> str:
    quantized_name, scale_name, zero_point_name, _ = projection_replacement(model, node)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    try:
        quantized = numpy_helper.to_array(initializers[quantized_name]).astype(np.float32)
        scale = numpy_helper.to_array(initializers[scale_name]).astype(np.float32)
        zero_point = numpy_helper.to_array(initializers[zero_point_name]).astype(np.float32)
    except KeyError as exc:
        raise SystemExit(f"Expected quantized weight initializers for {node.name}") from exc
    add_initializer_once(model, name, ((quantized - zero_point) * scale).astype(np.float32))
    return name


def current_projection_node(
    model: onnx.ModelProto,
    original_node: onnx.NodeProto,
    current_source: str,
    output_name: str,
    node_name: str,
    current_projection: str,
) -> onnx.NodeProto:
    if current_projection == "fp32":
        weight_name = dequantized_weight_initializer(model, original_node, f"{node_name}_weight_fp32")
        return helper.make_node("MatMul", [current_source, weight_name], [output_name], name=node_name)

    quantized_weight, weight_scale, weight_zero_point, _ = projection_replacement(model, original_node)
    return helper.make_node(
        "DynamicQuantizeMatMul",
        [current_source, quantized_weight, weight_scale, weight_zero_point, ""],
        [output_name],
        name=node_name,
        domain="com.microsoft",
    )


def prune_dead_nodes(model: onnx.ModelProto) -> None:
    producer = {output: index for index, node in enumerate(model.graph.node) for output in node.output}
    needed_values = {output.name for output in model.graph.output}
    needed_nodes: set[int] = set()
    pending = list(needed_values)

    while pending:
        value_name = pending.pop()
        node_index = producer.get(value_name)
        if node_index is None or node_index in needed_nodes:
            continue
        needed_nodes.add(node_index)
        for input_name in model.graph.node[node_index].input:
            if input_name and input_name not in needed_values:
                needed_values.add(input_name)
                pending.append(input_name)

    kept_nodes = [node for index, node in enumerate(model.graph.node) if index in needed_nodes]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    kept_value_info = [value_info for value_info in model.graph.value_info if value_info.name in needed_values]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)


def prune_unused_initializers(model: onnx.ModelProto) -> None:
    used = {input_name for node in model.graph.node for input_name in node.input if input_name}
    used.update(output.name for output in model.graph.output)
    kept = [initializer for initializer in model.graph.initializer if initializer.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def rewrite_model(input_path: Path, output_path: Path, current_projection: str) -> None:
    model = onnx.load(input_path)
    spec = infer_spec(model)
    if current_projection == "dynamic-int8" and not any(opset.domain == "com.microsoft" for opset in model.opset_import):
        model.opset_import.append(helper.make_operatorsetid("com.microsoft", 1))

    projection_names = {
        projection_node_name(layer, kind)
        for layer in range(spec.num_layers)
        for kind in ("k", "v")
    }
    projection_nodes = {name: find_node(model, name) for name in projection_names}
    remove_names: set[str] = set()
    for node in projection_nodes.values():
        remove_names.update(obsolete_projection_tail_names(model, node))

    for layer in range(spec.num_layers):
        for kind in ("key", "value"):
            add_graph_input_once(
                model,
                cache_layer_name(kind, layer),
                TensorProto.FLOAT,
                [spec.batch_size, spec.cache_len, spec.hidden_size],
            )
            add_graph_output_once(
                model,
                projected_current_name(kind, layer),
                TensorProto.FLOAT,
                [spec.batch_size, "current_frames", spec.hidden_size],
            )

    rewritten: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name not in remove_names:
            rewritten.append(node)
            continue
        if node.name not in projection_names:
            continue

        layer = layer_from_projection_node(node.name)
        is_key = "/linear_k/" in node.name
        kind = "key" if is_key else "value"
        short_kind = "k" if is_key else "v"
        prefix = layer_prefix(layer)
        current = projected_current_name(kind, layer)
        current_source = find_current_source(model, layer, node.input[0])
        _, _, _, output_name = projection_replacement(model, node)
        rewritten.append(
            current_projection_node(
                model,
                node,
                current_source,
                current,
                f"{prefix}/projected_current_{short_kind}_matmul",
                current_projection,
            )
        )
        rewritten.append(
            helper.make_node(
                "Concat",
                [cache_layer_name(kind, layer), current],
                [output_name],
                name=f"{prefix}/projected_{kind}_concat",
                axis=1,
            )
        )

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    prune_dead_nodes(model)
    prune_unused_initializers(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(
        f"[rewrite] wrote {output_path} layers={spec.num_layers} cacheLen={spec.cache_len} "
        f"hidden={spec.hidden_size} currentProjection={current_projection} nodes={len(model.graph.node)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--current-projection",
        choices=["fp32", "dynamic-int8"],
        default="dynamic-int8",
        help="Projection used for the current chunk before concatenating with projected cache.",
    )
    args = parser.parse_args()
    rewrite_model(args.input, args.output, args.current_projection)


if __name__ == "__main__":
    main()
