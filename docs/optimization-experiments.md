# Optimization Experiments

This log tracks ONNX/ORT optimization experiments ported from Sayboard's
Parakeet EOU work onto Wordpipe's Nemotron/Parakeet runtime.

See [sayboard-optimization-harvest.md](sayboard-optimization-harvest.md) for
the source-level Sayboard optimization inventory and remaining experiment queue.

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

## 2026-06-22: Conv Dequantization Ablation

Sayboard rejected conv quantization for its Parakeet EOU export. Wordpipe's
Nemotron export still contains quantized `ConvInteger` blocks after fixed-shape
specialization, so `scripts/dequantize_nemotron_conv_blocks.py` tests the
analogous rewrite: replace selected dynamic-quantized conv blocks with float
`Conv`, then serialize through ORT extended.

Applicability on `build/model-variants/nemotron-c56-fixed-shape/encoder.onnx`:

| Candidate family | Rewritable blocks | Encoder size |
| --- | ---: | ---: |
| `preconv_fp32` | 5 | 575 MiB |
| `allconv_fp32` | 77 | 792 MiB |

Build commands:

```sh
.venv/bin/python scripts/dequantize_nemotron_conv_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-preconv-fp32-ort \
  --include /pre_encode/conv/ \
  --ort-optimize-final extended \
  --ort-optimize-threads 1

.venv/bin/python scripts/dequantize_nemotron_conv_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-allconv-fp32-ort \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  fixed_shape_ort=build/model-variants/nemotron-c56-fixed-shape-ort-extended \
  preconv_fp32=build/model-variants/nemotron-c56-fixed-shape-preconv-fp32-ort \
  allconv_fp32=build/model-variants/nemotron-c56-fixed-shape-allconv-fp32-ort \
  --runs 3 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/conv-dequant-002.json
```

Settings:

- WAV: `build/allocation-ablation/librispeech-long.wav`
- Intra-op threads: `2`
- Flush chunks: `3`
- Runtime graph optimization flag: `all`
- Power profile: GNOME `balanced` at benchmark start and end. AC changed from
  offline to online during the run, but the profile stayed `balanced`.

Results:

| Variant | Encoder size | Median real-audio RTF | Delta vs `fixed_shape_ort` | Median decode seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_shape_ort` | 575 MiB | 0.703 | baseline | 86.877 | 10 / 313 = 3.19% |
| `preconv_fp32` | 575 MiB | 0.698 | +0.7% | 86.263 | 12 / 313 = 3.83% |
| `allconv_fp32` | 792 MiB | 0.637 | +9.4% | 78.753 | 12 / 313 = 3.83% |

Conv observations:

- Dequantizing only the five pre-encoder conv blocks is essentially runtime
  parity and worsens rough WER, so it is not attractive.
- Dequantizing all conv blocks is a real speed win on this benchmark, but it
  also worsens rough WER and previously changed the short smoke sample from
  "gold" to "code". Treat it as rejected unless a larger WER run shows that the
  regression is sample noise.
- The benchmark harness now records GNOME power-profiles-daemon state and can
  set a profile with `--set-power-profile`, which makes future runs more
  comparable than AC status alone. This daemon rejects `HoldProfile` for
  `balanced`, so the harness records that and falls back to set-and-verify for
  balanced-profile runs.

## 2026-06-22: FFN FP32 Thread-Count Sweep

The current best speed candidate, `ffn_fp32`, was benchmarked with ORT intra-op
threads set to 1, 2, 3, and 4 on the Ivy Bridge i5-3320M machine
(2 physical cores / 4 hardware threads). Each setting used three runs of the
long WAV benchmark under GNOME `balanced`.

Commands followed this pattern:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --runs 3 \
  --num-threads <N> \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/ffn-thread-sweep-t<N>.json
```

Results:

| ORT threads | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.879 | 0.866 | 108.632 | 112.647 |
| 2 | 0.606 | 0.597 | 74.930 | 78.985 |
| 3 | 0.860 | 0.848 | 106.333 | 110.653 |
| 4 | 0.820 | 0.808 | 101.338 | 105.483 |

Thread observations:

- The existing default, `--num-threads 2`, is the clear winner and should stay
  the default for this CPU.
- One thread underutilizes the two physical cores.
- Three and four threads are much worse than two threads, likely because this
  workload does not benefit from SMT oversubscription on the i5-3320M and pays
  extra scheduling/cache overhead.

## 2026-06-22: Live Audio Buffer Reuse

Sayboard's native bridge work emphasized avoiding avoidable allocation in the
hot recognition path. Wordpipe's CPAL live-input path still allocated a fresh
`Vec<f32>` in every audio callback before sending samples to the recognition
thread.

Change:

- Added a bounded audio buffer pool alongside the bounded audio queue in
  `wordpipe-parakeet-worker`.
- The CPAL callback now reuses a returned `Vec<f32>` when one is available,
  reserving only if the callback delivers a larger buffer than previously seen.
- The recognition thread recycles each buffer after extending the pending audio
  accumulator.

Validation:

```sh
cargo fmt
cargo check -p wordpipe-parakeet-worker
```

Notes:

- This targets live dictation stability and callback overhead, not long-WAV
  benchmark RTF. The benchmark path reads from a file and already reuses its
  chunk buffer.
- The bounded queue/drop behavior is unchanged: if the recognition thread falls
  behind, incoming callback buffers are still dropped and counted.

## 2026-06-22: ORT Session-Option Ablation

After thread count was settled at two intra-op threads, the worker exposed
several ORT session knobs for controlled benchmarking:

- `--ort-memory-pattern auto|enable|disable`
- `--ort-parallel-execution`
- `--ort-cpu-arena auto|enable|disable`

`auto` preserves the previous behavior and lets ORT choose its default. The
non-auto settings explicitly call the corresponding ORT session/CPU EP APIs.

Benchmark commands followed this pattern:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --runs 3 \
  --num-threads 2 \
  <ORT option under test> \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/ffn-ort-options-<variant>.json
```

Results:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds |
| --- | ---: | ---: | ---: | ---: |
| default / auto | 0.606 | 0.597 | 74.890 | 78.905 |
| `--ort-memory-pattern disable` | 0.630 | 0.621 | 77.944 | 81.948 |
| `--ort-memory-pattern enable` | 0.650 | 0.641 | 80.417 | 84.624 |
| `--ort-parallel-execution` | 0.644 | 0.635 | 79.627 | 83.893 |
| `--ort-cpu-arena disable` | 0.643 | 0.634 | 79.524 | 83.757 |
| `--ort-cpu-arena enable` | 0.610 | 0.602 | 75.477 | 79.548 |

Session-option observations:

- The existing ORT defaults are best on this machine.
- Explicit CPU arena enable is close to default but still slightly slower;
  forcing CPU EP registration is not useful here.
- Disabling the CPU arena is a throughput loss, even though it might reduce
  allocator memory retention in other conditions.
- Parallel graph execution and explicit memory-pattern overrides are both
  slower for this fixed-shape streaming graph.

## 2026-06-22: Current-Best ORT Profile

The current best speed candidate, `ffn_fp32`, was profiled directly through
ONNX Runtime with fixed synthetic streaming inputs. This isolates the encoder
graph from file I/O, decoder accumulation, and worker JSON output.

Command:

```sh
.venv/bin/python scripts/profile_nemotron_ort.py \
  --model-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --threads 2 \
  --graph-optimization all \
  --warmup 3 \
  --iterations 30 \
  --output-dir build/ort-profile/ffn-fp32-solo
```

Summary:

| Metric | Value |
| --- | ---: |
| Mean encoder run | 302.183 ms |
| Min encoder run | 290.339 ms |
| Max encoder run | 322.502 ms |
| Profiled node time | 9,601.885 ms |
| Provider | CPUExecutionProvider |

Top operation families by profiled node time:

| Op | Time |
| --- | ---: |
| `MatMul` | 2,309.615 ms |
| `FusedMatMul` | 2,187.441 ms |
| `ConvInteger` | 2,146.987 ms |
| `DynamicQuantizeMatMul` | 1,476.772 ms |
| `LayerNormalization` | 189.985 ms |
| `DynamicQuantizeLinear` | 161.063 ms |
| `Mul` | 155.017 ms |
| `Transpose` | 143.702 ms |

Profile observations:

- Full FFN dequantization did what we expected: the huge
  `DynamicQuantizeMatMul` block from the compact graph is mostly gone, replaced
  by FP32 `MatMul`/`FusedMatMul` work.
- The remaining major costs are now broad and structural: FP32 FFN matmuls,
  quantized convs, and the still-quantized non-FFN matmuls.
- All-conv FP32 was faster in the long-WAV benchmark but regressed rough WER, so
  the `ConvInteger` block is not a clean optimization target unless a larger
  validation run clears it.
- The remaining `DynamicQuantizeMatMul` cost is led by non-FFN nodes such as
  `/encoder/pre_encode/out/MatMul_quant`; attention-projection dequantization
  already tested poorly on rough WER.

### FFN Plus Pre-Encoder Output Dequantization

The ORT profile above showed `/encoder/pre_encode/out/MatMul_quant` as the
largest remaining non-FFN `DynamicQuantizeMatMul` site. The first attempt to
rewrite it from `build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort`
rewrote zero blocks because the source had already been serialized through ORT,
where the original `MatMulInteger` pattern was fused into
`DynamicQuantizeMatMul`. The valid procedure is to apply all selected
dequantization before the final ORT optimization pass.

`scripts/dequantize_nemotron_matmul_blocks.py` was adjusted so biased matmuls
whose input rank is unknown or not statically 2-D fall back to `MatMul` plus the
original `Add`, instead of emitting an invalid `Gemm`.

Build command:

```sh
.venv/bin/python scripts/dequantize_nemotron_matmul_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-preout-fp32-ort-v2 \
  --include /feed_forward \
  --include /pre_encode/out/ \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Result: `rewritten_blocks=97`, `prunedInitializers=194`, encoder size
`1739.6 MiB`.

Benchmark command:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  ffn_preout_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-preout-fp32-ort-v2 \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/ffn-preout-001.json
```

Results:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_fp32` | 0.685 | 0.675 | 84.710 | 89.447 | 9 / 313 = 2.88% |
| `ffn_preout_fp32` | 0.654 | 0.645 | 80.849 | 85.735 | 10 / 313 = 3.19% |

Observations:

- Adding the pre-encoder output projection to the FFN FP32 rewrite improved
  median real-audio RTF by about 4.5% relative to the same-session `ffn_fp32`
  baseline.
- The transcript changed from "had seemed" to "had seem", raising rough WER by
  one edit on the concatenated sample. Treat this as speed-positive but
  accuracy-suspicious until a broader validation run decides whether it is
  acceptable.
- This keeps `ffn_fp32` as the conservative current-best candidate, while
  `ffn_preout_fp32` is a follow-up WER-validation candidate.

## 2026-06-22: Nemotron Streaming Paper Leads

The paper at https://arxiv.org/html/2604.14493v1 is directly relevant because
it describes ONNX Runtime optimization for Nemotron Speech Streaming on
resource-constrained CPU devices.

Applicable learnings:

- Their selected streaming configuration is the same `(7, 10, 7)` chunk/history
  setting used by Wordpipe's c56 export: 560 ms chunks with 5.6 s history. This
  supports keeping the projected-cache rewrite aligned with that left context
  rather than shrinking cache length only for speed.
- Their quantization boundary matches our empirical direction: the encoder is
  the optimization target, while decoder/joiner stay FP32 because they are
  smaller and repeatedly invoked in the RNNT loop.
- Their strongest ONNX lead is weight-only block quantization using ORT
  `MatMulNBits`, with activations kept FP32. That is different from our current
  dynamic activation quantization path (`DynamicQuantizeLinear` plus
  `MatMulInteger`, often fused to `DynamicQuantizeMatMul`).
- Their `ConvInteger`/`MatMulInteger` result is a useful caution. It improves
  throughput but worsens WER, matching Wordpipe's all-conv FP32 ablation in the
  other direction: conv/integer arithmetic changes can be speed-positive while
  still being accuracy-risky.

Reported paper results for Nemotron Speech Streaming:

| Variant | Size | WER | RTFx |
| --- | ---: | ---: | ---: |
| FP32 ONNX | 2.47 GB | 8.03 | 6.73 |
| Int8 k-quant | 1.28 GB | 8.01 | 7.25 |
| Int4 k-quant | 0.67 GB | 8.20 | 7.20 |
| Int4 plus `ConvInteger`/`MatMulInteger` | 0.64 GB | 10.14 | 8.74 |

Next paper-derived experiments:

- Prototype an ORT `MatMulNBits` weight-only block-quantized encoder from the
  fixed-shape source, starting with int8 k-quant before int4.
- Compare against `ffn_fp32` and compact `fixed_shape_ort` with the same
  long-WAV 3-run median harness and the larger LibriSpeech sampled validation.
- Preserve decoder/joiner FP32 in that experiment.

### Int8 K-Quant MatMulNBits First Pass

`scripts/quantize_nemotron_matmul_nbits.py` wraps ONNX Runtime's lower-level
weight-only quantization implementation so the `MatMulNBits` experiment is
repeatable. The first pass targeted the current `ffn_fp32` candidate rather
than a fully FP32 pre-fusion encoder; this is intentionally narrow and should
not be read as a complete reproduction of the paper export.

Dry-run:

```sh
.venv/bin/python scripts/quantize_nemotron_matmul_nbits.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-nbits-int8-dryrun \
  --bits 8 \
  --block-size 32 \
  --algorithm k_quant \
  --dry-run
```

Result: 72 static-RHS `MatMul` nodes selected.

Build:

```sh
.venv/bin/python scripts/quantize_nemotron_matmul_nbits.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-nbits-int8-k32 \
  --bits 8 \
  --block-size 32 \
  --algorithm k_quant
```

Build output:

| Artifact | Value |
| --- | ---: |
| Selected `MatMul` nodes | 72 |
| Saved `MatMulNBits` nodes | 48 |
| Encoder size | 1.2 GiB |
| Baseline `ffn_fp32` encoder size | 1.7 GiB |

Benchmark:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  ffn_nbits_int8_k32=build/model-variants/nemotron-c56-fixed-shape-ffn-nbits-int8-k32 \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/ffn-nbits-int8-k32-001.json
```

Results:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_fp32` | 0.623 | 0.614 | 76.992 | 81.073 | 9 / 313 = 2.88% |
| `ffn_nbits_int8_k32` | 1.580 | 1.558 | 195.385 | 198.132 | 10 / 313 = 3.19% |

Observations:

- ORT 1.27's CPUExecutionProvider can execute `MatMulNBits` from the Rust
  worker, so the contrib-op path is functionally viable.
- This first pass is a throughput loss on the Ivy Bridge i5-3320M despite
  reducing encoder size by about 30%. It is also one edit worse on the rough
  concatenated sample.
- The likely reason is that this path quantized a partially optimized mixed
  graph: 99 `DynamicQuantizeMatMul`, 48 `FusedMatMul`, and 72 ordinary
  `MatMul` nodes remained alongside 48 `MatMulNBits` nodes. It did not recreate
  the paper's encoder-wide FP32-to-weight-only export.
- Next `MatMulNBits` attempts, if any, should start earlier in the pipeline:
  export/fold/fix shapes while preserving ordinary static-RHS `MatMul` nodes,
  then apply weight-only quantization before ORT fuses the graph.

### Pre-Fusion FFN MatMulNBits

The post-ORT `MatMulNBits` pass above only produced 48 saved `MatMulNBits`
nodes because much of the graph had already been fused. To test whether
pre-fusion placement changes the result, a non-ORT-serialized FFN-FP32 source
was built first:

```sh
.venv/bin/python scripts/dequantize_nemotron_matmul_blocks.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-preort \
  --include /feed_forward
```

Result: `rewritten_blocks=96`, `prunedInitializers=192`, `ortFinal=None`,
encoder size `1779.0 MiB`. A dry-run from this source selected 96 static-RHS
FFN `MatMul` nodes for weight-only quantization.

The low-level k-quant path completed quantization but produced an invalid graph
for this pre-fusion source: ONNX checker reported a missing producer for
`/encoder/pre_encode/conv/conv.0/Conv_output_0_bias_reshape_output`. This
appears to be a limitation or bug in the neural-compressor-derived ORT helper
on this graph, so the wrapper was extended to use ORT's newer
`MatMulNBitsQuantizer` default path as `--algorithm default`.

Build:

```sh
.venv/bin/python scripts/quantize_nemotron_matmul_nbits.py \
  --source-dir build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-preort \
  --output-dir build/model-variants/nemotron-c56-fixed-shape-ffn-nbits-int8-default-k32-preort \
  --bits 8 \
  --block-size 32 \
  --algorithm default
```

Build output:

| Artifact | Value |
| --- | ---: |
| Selected `MatMul` nodes | 96 |
| Saved `MatMulNBits` nodes | 96 |
| Remaining `DynamicQuantizeMatMul` nodes | 48 |
| Remaining ordinary `MatMul` nodes | 72 |
| Encoder proto | 1.9 MiB |
| Encoder external data | 685.2 MiB |

Benchmark:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  ffn_nbits_int8_default_preort=build/model-variants/nemotron-c56-fixed-shape-ffn-nbits-int8-default-k32-preort \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/ffn-nbits-int8-default-preort-001.json
```

The benchmark produced three baseline `ffn_fp32` runs but the first candidate
run timed out after the 300 second per-run limit. The partial result file is
`build/parakeet-variant-bench/ffn-nbits-int8-default-preort-001.json`.

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Result |
| --- | ---: | ---: | ---: | --- |
| `ffn_fp32` | 0.616 | 0.607 | 76.183 | baseline median |
| `ffn_nbits_int8_default_preort` | n/a | n/a | >300 | timed out on run 1 |

Observations:

- Applying `MatMulNBits` before ORT fusion does create the intended 96 FFN
  weight-only nodes and reduces the model size substantially.
- Throughput is still decisively worse on this CPU. The candidate failed to
  finish a single long-WAV run within 300 seconds, versus a 76.183 second
  baseline median decode.
- This rejects the FFN-only `MatMulNBits` path for the current Ivy Bridge CPU
  and ORT 1.27 CPU EP. The paper's full encoder-wide export may still behave
  differently on newer CPUs or with a different ORT build, but the practical
  Wordpipe path remains `ffn_fp32`.

## 2026-06-22: FP32 NeMo Export With Projected Cache

The projected-cache rewrite originally targeted the quantized sherpa-style
encoder graph. To test whether the same cache rewrite was applicable earlier in
the NeMo export pipeline, `scripts/rewrite_nemotron_projected_kv_cache.py` was
extended to support native FP32 `MatMul` K/V projection nodes and external-data
serialization. `scripts/build_nemotron_fixed_shape_model.py` was also updated
to save/check large external-data models by path so FP32 graphs do not hit
ONNX's in-memory >2 GB checker path.

Build from the existing interrupted FP32 export artifacts:

```sh
.venv/bin/python scripts/transform_nemotron_parakeet_export.py \
  build/model-variants/nemotron-fp32-projected \
  --no-quantize \
  --projected-cache \
  --keep-fp32
```

Fixed-shape and ORT-optimized build:

```sh
.venv/bin/python scripts/build_nemotron_fixed_shape_model.py \
  --source-dir build/model-variants/nemotron-fp32-projected \
  --output-dir build/model-variants/nemotron-fp32-projected-fixed-shape \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Resulting fixed-shape artifact:

| Artifact | Size |
| --- | ---: |
| `encoder.onnx` | 12.2 MiB |
| `encoder.onnx.data` | 2340.7 MiB |
| `decoder_joint.onnx` | 93.1 MiB |

Benchmark:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  ffn_fp32=build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort \
  fp32_projected=build/model-variants/nemotron-fp32-projected-fixed-shape \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/fp32-projected-001.json
```

The system reported `BAT0/status=Discharging` for this run even though GNOME's
power profile was pinned to `balanced`, so compare it as a same-run A/B result
rather than a clean AC-powered absolute benchmark.

WER scoring:

```sh
.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/fp32-projected-001.json
```

Results:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_fp32` | 0.666 | 0.656 | 82.289 | 87.045 | 9 / 313 = 2.88% |
| `fp32_projected` | 0.590 | 0.582 | 73.013 | 76.506 | 11 / 313 = 3.51% |

Observations:

- The FP32 NeMo export plus projected cache is throughput-positive in this
  A/B: median real-audio RTF improved by about 11.4% versus `ffn_fp32`.
- It fails the WER gate. The transcript changed consistently across all three
  runs, including "had seemed" -> "had seem", raising rough WER from 9/313 to
  11/313.
- Do not promote this as the default runtime candidate. The useful harvest is
  the hardened projected-cache/export tooling and the reusable
  `scripts/score_benchmark_wer.py` WER check.

### Raw-Cache Control

To isolate whether the WER regression came from projected-cache rewriting or
from the FP32 NeMo export path itself, the same consolidated FP32 encoder was
also transformed without projected cache and then built with the same
fixed-shape/ORT-extended path.

Build:

```sh
.venv/bin/python scripts/transform_nemotron_parakeet_export.py \
  build/model-variants/nemotron-fp32-rawcache \
  --no-quantize \
  --no-projected-cache \
  --keep-fp32

.venv/bin/python scripts/build_nemotron_fixed_shape_model.py \
  --source-dir build/model-variants/nemotron-fp32-rawcache \
  --output-dir build/model-variants/nemotron-fp32-rawcache-fixed-shape \
  --ort-optimize-final extended \
  --ort-optimize-threads 1
```

Benchmark:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  fp32_rawcache=build/model-variants/nemotron-fp32-rawcache-fixed-shape \
  fp32_projected=build/model-variants/nemotron-fp32-projected-fixed-shape \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/fp32-rawcache-isolation-001.json
```

WER scoring:

```sh
.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/fp32-rawcache-isolation-001.json
```

Results:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp32_rawcache` | 0.687 | 0.677 | 84.911 | 88.036 | 11 / 313 = 3.51% |
| `fp32_projected` | 0.516 | 0.509 | 63.825 | 67.981 | 11 / 313 = 3.51% |

The system again reported battery discharging (`25%` -> `18%`) with GNOME
profile `balanced`, so use this as an in-run A/B only.

Conclusion:

- The WER regression is not caused by the projected-cache rewrite. Raw-cache
  FP32 and projected-cache FP32 produce the same transcript and the same rough
  WER.
- Projected cache is doing what we expected on this export: it removes the cost
  of repeatedly projecting the full raw K/V cache, improving median real-audio
  RTF by about 24.9% versus raw-cache FP32.
- The remaining accuracy question is why this FP32 NeMo export path differs
  from the sherpa-derived `ffn_fp32` candidate (`11/313` vs `9/313` on this
  rough sample).

## 2026-06-22: Sayboard Harvest Wrapper Results

The remaining Sayboard-derived experiments are now captured by:

```sh
.venv/bin/python scripts/run_sayboard_harvest_experiments.py --force
```

The wrapper rebuilds the release worker before benchmarking, because stale
worker binaries can otherwise hide ABI changes such as removing
`processed_signal_length` from the encoder inputs.

Fixed-length ABI result:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.641 | 0.631 | 79.202 | 83.357 | 9 / 313 = 2.88% |
| `fixed_length` | 0.706 | 0.696 | 87.338 | 88.534 | 10 / 313 = 3.19% |

Result file:
`build/parakeet-variant-bench/sayboard-fixed-length-001.json`

FP32 current-projection result:

| Variant | Median real-audio RTF | Median RTF | Median decode seconds | Median wall seconds | Rough WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.642 | 0.632 | 79.330 | 83.529 | 9 / 313 = 2.88% |
| `fp32_current_projection` | 0.729 | 0.718 | 90.114 | 91.732 | 12 / 313 = 3.83% |

Result file:
`build/parakeet-variant-bench/sayboard-fp32-current-projection-001.json`

Conclusion:

- Removing `processed_signal_length` is not a win for the current ORT CPU path.
  It made the graph smaller but produced slower decoding and a small WER
  regression on this sample.
- FP32 current K/V projection after quantization is also not a win here. It is
  slower than the current `ffn_fp32` baseline and inherits/worsens the FP32
  export accuracy gap.
- Keep `build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort` as the
  current best local candidate.
