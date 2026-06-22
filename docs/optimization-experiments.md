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

- Evaluate `linear_pos_fp32` on the existing LibriSpeech manifest before keeping
  it.
- Generate an `ort_extended` deployable model directory from the source model as
  part of the transform pipeline.
- Test fixed-shape/static-shape folding only after confirming it survives as a
  runtime win beyond ORT's session-time optimization.
