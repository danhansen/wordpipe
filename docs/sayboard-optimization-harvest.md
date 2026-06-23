# Sayboard Optimization Harvest

This document tracks the ONNX/ORT-relevant optimizations found in
`/home/dhansen/Downloads/Sayboard` and their Wordpipe status. The goal is to
avoid losing useful ideas while keeping each candidate tied to a repeatable
benchmark and WER check.

Benchmark convention for Wordpipe experiments:

```sh
.venv/bin/python scripts/benchmark_parakeet_variant.py \
  candidate=path/to/model \
  --runs 3 \
  --num-threads 2 \
  --min-mem-available-gb 6 \
  --child-memory-limit-gb 10 \
  --set-power-profile balanced \
  --output build/parakeet-variant-bench/name.json

.venv/bin/python scripts/score_benchmark_wer.py \
  build/parakeet-variant-bench/name.json
```

## Current Wordpipe Default Candidate

The current conservative speed candidate remains
`build/model-variants/nemotron-c56-fixed-shape-ffn-fp32-ort`:

- projected K/V cache;
- fixed c56 streaming shapes;
- ORT `extended` serialized encoder;
- feed-forward MatMul/Gemm blocks dequantized back to FP32.

On the concatenated LibriSpeech WAV, this candidate has repeatedly scored
`9 / 313 = 2.88%` rough WER. Throughput varies with power state, so compare
only within the same benchmark run.

## Harvest Matrix

| Sayboard optimization | Sayboard source | Wordpipe status | Evidence / next action |
| --- | --- | --- | --- |
| Deterministic LibriSpeech ablation protocol with timing and WER | `scripts/parakeet_ablation/run_ablation.py`, `README.md` | Ported | `scripts/benchmark_parakeet_variant.py`, `scripts/eval_librispeech_backends.py`, and `scripts/score_benchmark_wer.py` provide the long-WAV 3-run median plus rough WER convention. |
| Fixed streaming shapes | `rewrite_fixed_streaming_shapes.py` | Ported and kept | `scripts/build_nemotron_fixed_shape_model.py`; documented speed win in `docs/optimization-experiments.md`. |
| Resolve symbolic dims and replace static `Shape` nodes | `rewrite_fixed_streaming_shapes.py`, `sanitize_tflite_encoder.py` | Ported and kept | Fixed-shape build resolves known dims and replaces static `Shape` nodes. The effect is folded into current default builds. |
| ORT serialized graph optimization | `sanitize_tflite_encoder.py` uses `basic` for converter readiness; Sayboard runtime uses ORT/LiteRT sessions | Ported and benchmarked | Wordpipe tested runtime levels/session options. ORT `extended` serialized fixed-shape graph is the current default; `basic` is useful mainly for converter-safe folding. |
| Projected K/V cache | `rewrite_projected_kv_cache.py` | Ported and kept | Wordpipe supports projected-cache graphs and runtime cache rolling. Raw-cache FP32 control showed same WER as projected-cache FP32, with projected cache much faster. |
| Layered projected-cache ABI | `rewrite_projected_kv_cache.py` | Ported and kept | Wordpipe uses per-layer `cache_key_layer_N` / `cache_value_layer_N` inputs and `projected_current_*` outputs, matching Sayboard's layered ABI. |
| Stacked projected-cache ABI | `rewrite_projected_kv_cache.py` | Not ported | Not useful for Wordpipe today. Layered ABI avoids in-graph cache rolling and matches the Rust runtime. |
| FP32 current K/V projection after quantized source graph | `build_deployed_model.py` calls projected-cache rewrite with `current_projection="fp32"` | Implemented; benchmark pending | Wordpipe default projected-cache quantized graph uses `dynamic-int8` current projection, but export/transform scripts now accept `--projected-cache-current-projection fp32`. Build and benchmark/WER this variant. |
| Dynamic int8 quantization by operator family | `run_ablation.py` default variants | Partially ported and benchmarked | Wordpipe tested broad quantized baseline, FFN dequantization, pre-encoder output dequantization, conv dequantization, layer slices/even/odd variants, and MatMulNBits. Current best dequantizes FFN blocks back to FP32. |
| Per-channel dynamic quantization sweeps | `run_ablation.py` `*_pc` variants | Mostly untested in Wordpipe | Worth testing only from a clean FP32 export or targeted re-quantization path. Current sherpa-derived source is already quantized, so per-channel variants are not a simple post-pass on the default artifact. |
| Dynamic Conv quantization | `README.md` rejected conv variants | Ported and rejected | Wordpipe `scripts/quantize_nemotron_conv_dynamic.py` and conv dequant experiments showed throughput/accuracy tradeoffs were not attractive. |
| MatMul/Gemm dynamic quantization from fixed raw-cache FP32 | `build_deployed_model.py`, `run_ablation.py` `fullpre_*` variants | Partially tested | Wordpipe's FP32 NeMo raw-cache/projected-cache controls are functional, but the FP32 export path scored worse WER than the sherpa-derived candidate. Do not use as default without explaining the export parity gap. |
| Remove fixed length input and replace with initializer | `rewrite_fixed_streaming_shapes.py --keep-length-input` default removes `length` | Implemented; benchmark pending | Wordpipe can now build this ABI with `scripts/build_nemotron_fixed_shape_model.py --constant-processed-signal-length`, and the Rust runtime feeds `processed_signal_length` only when the encoder exposes it. Needs same-WAV 3-run benchmark plus WER scoring. |
| MatMulInteger quantization tail to FP32 `Gemm` cleanup | `rewrite_quantized_matmulinteger_to_gemm.py` | Not directly applicable as an ORT speed optimization | Sayboard used this before TFLite conversion. For ORT it intentionally dequantizes quantized blocks, overlapping with Wordpipe's targeted FFN FP32 dequantization but too broad for the default path. |
| TFLite/LiteRT conversion and static-RHS BMM to FC rewrite | `scripts/litert_spike/*` | Out of current Linux ORT scope | Useful if Wordpipe later adds a LiteRT backend. Sayboard's own notes show ORT was faster than host LiteRT FP32 for the simple encoder, while Android/device results were the main motivation. |
| Custom Android ORT with NCHWc/NEON | `build_onnxruntime_android_nchwc.sh` | Not applicable to Linux x86_64 | The analogous Wordpipe path is `scripts/build_onnxruntime_ivybridge.sh`, but current Python/Rust ORT binaries already execute acceptably and custom builds are deferred. |
| Runtime thread count sweep | Rust bridge config defaults and Wordpipe benchmark harness | Ported and benchmarked | Wordpipe tested 1-4 intra-op threads; 2 threads remains the practical benchmark default for this CPU. |
| ORT memory pattern / arena / parallel execution toggles | Wordpipe follow-up inspired by runtime tuning | Ported and benchmarked | `scripts/benchmark_parakeet_variant.py` exposes these toggles. Defaults remained best or close enough; explicit parallel execution was worse. |
| Allocation reduction in Rust runtime | Sayboard Rust bridge and Wordpipe fork changes | Mostly ported | Wordpipe runtime updates caches in place and reuses audio buffers. Prior A/B showed small or noisy gains; no remaining obvious Sayboard allocation trick is unported for the Nemotron path. |

## Pending Experiments

These are the remaining ONNX/ORT-relevant Sayboard ideas that are plausible for
Wordpipe and still lack a same-WAV 3-run median plus WER result:

1. Quantized projected-cache with FP32 current K/V projection.
   Build with `--projected-cache-current-projection fp32`, apply
   fixed-shape/ORT extended, then benchmark against the current `ffn_fp32`
   default and score WER. The export/transform CLI support is implemented;
   only the heavy run is pending.

2. Fixed `processed_signal_length` initializer.
   Build with `--constant-processed-signal-length` and benchmark the resulting
   variant against the current default. The builder/runtime ABI support is
   implemented; only the heavy run is pending because current machine power is
   low.

3. Per-channel quantization from a clean FP32 source.
   Only run this if the FP32 export parity issue is resolved or if we can
   reconstruct a sherpa-equivalent FP32 source. Applying per-channel dynamic
   quantization after the current sherpa-derived int8 export is not the same
   experiment.

4. Export parity investigation.
   The FP32 NeMo raw-cache and projected-cache controls both scored `11 / 313`
   WER while the sherpa-derived `ffn_fp32` candidate scores `9 / 313`. Before
   promoting any FP32-export-derived optimization, compare graph inputs,
   preprocessing assumptions, prompt/lang handling, and decoder/joint export
   parity against the sherpa-derived model.

## Rejected Or Deferred

- Broad `MatMulNBits` on this Ivy Bridge CPU: model size improves, but
  throughput was much worse and WER did not improve.
- Conv quantization/dequantization variants: not attractive on the current
  rough WER sample.
- Full TFLite/LiteRT backend: outside the current Linux Wayland/GNOME ORT
  runtime goal, and Sayboard's host notes do not show an obvious Linux CPU win.
- Stacked projected-cache ABI: more graph-side cache rolling with no current
  runtime benefit.
