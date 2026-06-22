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

## 2026-06-22: Fixed-Shape MatMul Dequantization Ablation

Sayboard's Parakeet EOU ablation suite tested selective dynamic-int8 MatMul/Gemm
quantization. For Wordpipe's already-quantized Nemotron export, the analogous
experiment is to rewrite selected static-RHS `MatMulInteger` blocks back to
float `MatMul`/`Gemm`, then serialize the final graph through ORT extended.

Applicability check on `build/model-variants/nemotron-c56-fixed-shape/encoder.onnx`:

| Candidate family | Rewritable blocks | Note |
| --- | ---: | --- |
| `linear_pos_fp32` | 1 | Shared relative-position projection block |
| `attn_proj_fp32` | 48 | Self-attention Q/K/V/output projections |
| `ffn_fp32` | 96 | Feed-forward MatMul/Gemm blocks |
| attention score/context | 0 | Dynamic-dynamic attention matmuls have no static RHS to dequantize |

Build commands:

```sh
.venv/bin/python scripts/dequantize_nemotron_matmul_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-linear-pos-fp32-ort \
  --include /self_attn/linear_pos/ \
  --ort-optimize-final extended \
  --ort-optimize-threads 1

.venv/bin/python scripts/dequantize_nemotron_matmul_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-attn-proj-fp32-ort \
  --include /self_attn/linear_q/ \
  --include /self_attn/linear_k/ \
  --include /self_attn/linear_v/ \
  --include /self_attn/linear_out/ \
  --ort-optimize-final extended \
  --ort-optimize-threads 1

.venv/bin/python scripts/dequantize_nemotron_matmul_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --include /feed_forward \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Artifact sizes:

| Variant | Encoder size |
| --- | ---: |
| `fixed_shape_ort` | 575 MiB |
| `linear_pos_fp32` | 575 MiB |
| `attn_proj_fp32` | 719 MiB |
| `ffn_fp32` | 1.7 GiB |

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  fixed_shape_ort=build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  linear_pos_fp32=build/model-variants/nemotron-c56-fixed-shape-linear-pos-fp32-ort \
  attn_proj_fp32=build/model-variants/nemotron-c56-fixed-shape-attn-proj-fp32-ort \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --runs 3 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --output build/parakeet-variant-bench/fixed-shape-dequant-001.json
```

Settings:

- WAV: `build/allocation-ablation/librispeech-long.wav`
- Intra-op threads: `2`
- Flush chunks: `3`
- Runtime graph optimization flag: `all`
- Memory guard: each run required at least 6 GiB `MemAvailable`; each worker
  subprocess had `RLIMIT_AS=10 GiB`.
- Memory metadata: start `MemAvailable=11.1 GiB`, swap used `3.1 GiB`; end
  `MemAvailable=11.6 GiB`, swap used `2.9 GiB`.

Results:

| Variant | Median RTF | Median real-audio RTF | Median decode seconds | Delta vs `fixed_shape_ort` |
| --- | ---: | ---: | ---: | ---: |
| `fixed_shape_ort` | 0.709 | 0.719 | 88.922 | baseline |
| `linear_pos_fp32` | 0.827 | 0.839 | 103.700 | -16.7% |
| `attn_proj_fp32` | 0.710 | 0.720 | 89.011 | -0.1% |
| `ffn_fp32` | 0.618 | 0.627 | 77.506 | +12.8% |

Rough concatenated-reference WER, using the current
`build/librispeech-backend-eval/manifest.jsonl` parser:

| Variant | Edits / words | WER |
| --- | ---: | ---: |
| `fixed_shape_ort` | 10 / 313 | 3.19% |
| `linear_pos_fp32` | 12 / 313 | 3.83% |
| `attn_proj_fp32` | 12 / 313 | 3.83% |
| `ffn_fp32` | 9 / 313 | 2.88% |

Graph diagnostics for `ffn_fp32`:

| Variant | Nodes | `MatMul` | `DynamicQuantizeMatMul` | `ConvInteger` | Float initializer bytes | UINT8 initializer bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_fp32` | 2,235 | 120 | 99 | 77 | 1549.1 MiB | 176.8 MiB |

Observations:

- `linear_pos_fp32` does not carry over as a win after fixed-shape folding. It
  is slower and worsens the rough WER in this run set.
- `attn_proj_fp32` is effectively runtime parity but larger and changes text,
  so it is not attractive by itself.
- `ffn_fp32` is the first selective dequantization that moves the performance
  needle after fixed-shape specialization: +12.8% real-audio RTF versus
  `fixed_shape_ort`, with a slightly better rough WER on this concatenated
  sample. The tradeoff is model size: 1.7 GiB for the encoder.
- The likely explanation is CPU-kernel economics: for these large FFN
  static-RHS matmuls, ORT's FP32 GEMM path beats dynamic activation
  quantization plus int8 matmul overhead on this machine.

Next candidates:

- Validate `ffn_fp32` on the larger LibriSpeech sampled evaluation before
  promoting it as the runtime model.
- Consider an intermediate FFN subset ablation by layer if the 1.7 GiB model
  size is too expensive.
- Keep `fixed_shape_ort` as the compact default candidate until the larger WER
  run confirms `ffn_fp32`.

### LibriSpeech Sample Validation

The existing 9-utterance LibriSpeech manifest was rerun with per-utterance
worker subprocesses and the same memory safety guard:

```sh
.venv/bin/python scripts/eval_librispeech_backends.py \
  --manifest build/librispeech-backend-eval/manifest.jsonl \
  --work-dir build/librispeech-ffn-validation/fixed-shape-ort \
  --backend parakeet \
  --parakeet-model-dir build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10

.venv/bin/python scripts/eval_librispeech_backends.py \
  --manifest build/librispeech-backend-eval/manifest.jsonl \
  --work-dir build/librispeech-ffn-validation/ffn-fp32 \
  --backend parakeet \
  --parakeet-model-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10
```

Results:

| Variant | Samples | WER | Decode RTF | Real-audio decode RTF | Wall RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_shape_ort` | 9 | 9 / 313 = 2.88% | 0.707 | 0.807 | 0.921 |
| `ffn_fp32` | 9 | 9 / 313 = 2.88% | 0.610 | 0.697 | 1.017 |

Validation observations:

- `ffn_fp32` preserved aggregate WER on this sampled LibriSpeech set and
  improved real-audio decode RTF by about 13.7%.
- Wall RTF was worse for `ffn_fp32` in this eval harness because each utterance
  starts a fresh worker process/session and the `ffn_fp32` encoder is much
  larger. The long-WAV benchmark is the better proxy for a persistent dictation
  worker.
- This moves `ffn_fp32` from "interesting" to "candidate", with model size and
  session-load behavior as the remaining tradeoffs to quantify.

### FFN Slice Ablation

To test whether the 1.7 GiB full-FFN artifact could be reduced, four 48-block
FFN variants were built from the same fixed-shape source:

- `ffn_early_fp32`: layers 0-11 feed-forward blocks.
- `ffn_late_fp32`: layers 12-23 feed-forward blocks.
- `ffn_even_fp32`: even-numbered layer feed-forward blocks.
- `ffn_odd_fp32`: odd-numbered layer feed-forward blocks.

Each slice rewrote 48 blocks, pruned 96 now-unused initializers, and produced a
1.2 GiB encoder.

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  fixed_shape_ort=build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  ffn_early_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-early-fp32-ort \
  ffn_late_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-late-fp32-ort \
  ffn_even_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-even-fp32-ort \
  ffn_odd_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-odd-fp32-ort \
  --runs 3 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --output build/parakeet-variant-bench/ffn-slices-002.json
```

The benchmark harness now writes checkpoint JSON after each completed run, so
long ablation batches leave `status=partial` evidence if interrupted.

Results:

| Variant | Encoder size | Median real-audio RTF | Delta vs `fixed_shape_ort` | Rough WER |
| --- | ---: | ---: | ---: | ---: |
| `fixed_shape_ort` | 575 MiB | 0.708 | baseline | 10 / 313 = 3.19% |
| `ffn_fp32` | 1.7 GiB | 0.608 | +14.1% | 9 / 313 = 2.88% |
| `ffn_early_fp32` | 1.2 GiB | 0.655 | +7.5% | 10 / 313 = 3.19% |
| `ffn_late_fp32` | 1.2 GiB | 0.657 | +7.2% | 10 / 313 = 3.19% |
| `ffn_even_fp32` | 1.2 GiB | 0.655 | +7.5% | 11 / 313 = 3.51% |
| `ffn_odd_fp32` | 1.2 GiB | 0.676 | +4.5% | 10 / 313 = 3.19% |

Slice observations:

- Half-FFN dequantization gives roughly half the full-FFN speedup, not most of
  it. The runtime benefit appears fairly proportional to the number of FFN
  blocks moved back to FP32.
- Early and late contiguous halves are equivalent on this benchmark and preserve
  rough WER. Even is not attractive because rough WER worsened; odd is slower.
- Full `ffn_fp32` remains the best speed candidate. The half variants are
  fallback compromises only if the 1.7 GiB encoder is unacceptable.
