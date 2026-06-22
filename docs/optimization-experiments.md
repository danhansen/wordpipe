# Optimization Experiments

This log tracks ONNX/ORT optimization experiments ported from Sayboard's
Parakeet EOU work onto Wordpipe's Nemotron/Parakeet runtime.

## 2026-06-22: ORT Serialization And Linear-Pos Dequantization

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  baseline=models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56 \
  ort_extended=build/model-variants/nemotron-c56-ort-extended \
  linear_pos_fp32=build/model-variants/nemotron-c56-linear-pos-fp32 \
  --runs 3 \
  --output build/parakeet-variant-bench/onnx-ort-harvest-001.json
```

Settings:

- WAV: `build/allocation-ablation/librispeech-long.wav`
- Intra-op threads: `2`
- Flush chunks: `3`
- Runtime graph optimization flag: `all`

Results:

| Variant | Median RTF | Median real-audio RTF | Median decode seconds | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 1.028 | 1.043 | 128.970 | baseline |
| `ort_extended` | 0.883 | 0.896 | 110.745 | +14.1% decode speed |
| `linear_pos_fp32` | 0.877 | 0.890 | 110.072 | +14.7% decode speed |

Rough concatenated-reference WER, using
`build/librispeech-backend-eval/manifest.jsonl` as the source transcript for the
long WAV:

| Variant | Edits / words | WER |
| --- | ---: | ---: |
| `baseline` | 9 / 310 | 2.90% |
| `ort_extended` | 9 / 310 | 2.90% |
| `linear_pos_fp32` | 10 / 310 | 3.23% |

Observations:

- Serializing ORT's `extended` optimized encoder graph is a clean win in this
  run set. It produced the same transcript as baseline and reduced median
  decode time by about 14%.
- The targeted `linear_pos_fp32` rewrite was slightly faster than the serialized
  ORT variant, but it changed decoded text and slightly worsened the rough
  concatenated-reference WER in this run set.
- The `linear_pos_fp32` script rewrites one source `self_attn/linear_pos`
  quantized matmul. ORT expands/fuses the graph later, so the single source
  block corresponds to the repeated hot `linear_pos` kernels seen in ORT
  profiling.

Next candidates:

- Build future model directories with `--ort-optimize-final extended` in either
  `scripts/transform_nemotron_parakeet_export.py` or
  `scripts/export_nemotron_parakeet_optimized.py`; this is the clean candidate
  from this run set.
- Evaluate `linear_pos_fp32` on a larger LibriSpeech manifest before keeping it.
- Test fixed-shape/static-shape folding only after confirming it survives as a
  runtime win beyond ORT's session-time optimization.

## 2026-06-22: Fixed Streaming Shapes

Sayboard's Parakeet EOU export path specialized the runtime ONNX graph around
the actual streaming chunk/cache shapes. `scripts/build_nemotron_fixed_shape_model.py`
ports that idea to the Nemotron/Parakeet export by:

- fixing streaming input/output shapes for the current c56 model:
  `[1, 128, 65]` mel chunks, `[24, 1, 56, 1024]` channel cache,
  `[24, 1, 1024, 8]` time cache, and `[1, 1024, 7]` encoded output;
- resolving symbolic dimensions where those fixed values are known;
- replacing static `Shape` nodes with int64 initializers;
- optionally serializing ORT's final optimized graph with
  `--ort-optimize-final extended`.

Build commands:

```sh
.venv/bin/python scripts/build_nemotron_fixed_shape_model.py \
  --source-dir models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56 \
  --output-dir build/model-variants/nemotron-c56-fixed-shape

.venv/bin/python scripts/build_nemotron_fixed_shape_model.py \
  --source-dir models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56 \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Graph diagnostics:

| Variant | Nodes | Unresolved dims | `Shape` | `Gather` | `MatMulInteger` | `DynamicQuantizeMatMul` | `MatMulIntegerToFloat` | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ort_extended` | 2,981 | 4,660 | 111 | 211 | 0 | 195 | 24 | 626.3 MiB |
| `fixed_shape` source | 6,893 | 4,138 | 231 | n/a | n/a | n/a | n/a | 627.1 MiB |
| `fixed_shape` after ORT extended | 2,091 | 222 | 0 | 0 | 0 | 195 | 0 | 574.8 MiB |
| `fixed_shape_ort` serialized | 2,091 | 222 | 0 | 0 | 0 | 195 | 0 | 574.8 MiB |

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ort_extended=build/model-variants/nemotron-c56-ort-extended \
  fixed_shape=build/model-variants/nemotron-c56-fixed-shape \
  fixed_shape_ort=build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  --runs 3 \
  --output build/parakeet-variant-bench/fixed-shape-ac-002.json
```

Settings:

- WAV: `build/allocation-ablation/librispeech-long.wav`
- Intra-op threads: `2`
- Flush chunks: `3`
- Runtime graph optimization flag: `all`
- Power metadata: AC online for all runs; battery charged from 29% to 45%;
  governor `schedutil`; turbo not disabled.

Results from this AC batch:

| Variant | Median RTF | Median real-audio RTF | Median decode seconds | Delta vs `ort_extended` |
| --- | ---: | ---: | ---: | ---: |
| `ort_extended` | 1.236 | 1.253 | 154.987 | baseline |
| `fixed_shape` | 0.889 | 0.902 | 111.493 | +28.0% real-audio RTF |
| `fixed_shape_ort` | 0.825 | 0.837 | 103.440 | +33.2% real-audio RTF |

Rough concatenated-reference WER, using the current
`build/librispeech-backend-eval/manifest.jsonl` parser:

| Variant | Edits / words | WER |
| --- | ---: | ---: |
| `ort_extended` | 10 / 313 | 3.19% |
| `fixed_shape` | 10 / 313 | 3.19% |
| `fixed_shape_ort` | 10 / 313 | 3.19% |

Observations:

- Fixed streaming shapes are a larger win than plain ORT serialization in this
  setup. They remove runtime shape plumbing that ORT could not eliminate from
  the symbolic export and allow the final optimized graph to drop from 2,981
  nodes to 2,091 nodes.
- The serialized fixed-shape graph is the best candidate artifact from this
  run set: it keeps the same transcript/WER as `ort_extended`, is smaller on
  disk, avoids session-time graph optimization work, and was fastest in the AC
  batch.
- Absolute RTF varied by machine state during the day. The benchmark harness now
  records AC, battery, governor, CPU frequency, turbo, and platform profile
  metadata so future timing JSON can be audited.

Next candidates:

- Prefer `build/model-variants/nemotron-c56-fixed-shape-ort-extended` as the
  current runtime model candidate.
- Add randomized variant order or warmup support to the benchmark harness if we
  need smaller deltas than this fixed-shape result.
- Continue looking for Sayboard optimizations beyond fixed-shape specialization
  only after this artifact is validated against a larger LibriSpeech sample.
