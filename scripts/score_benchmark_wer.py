#!/usr/bin/env python3
"""Score rough WER and matching RTF for benchmark_parakeet_variant.py output.

The long-WAV benchmark concatenates a small LibriSpeech manifest into one WAV
and stores the final transcript and worker metrics for each run. This script
reconstructs that concatenated reference, applies the same normalization/edit
distance helpers used by eval_librispeech_backends.py, and reports speed and
accuracy from the same benchmark rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_eval_helpers():
    path = Path(__file__).resolve().parent / "eval_librispeech_backends.py"
    spec = importlib.util.spec_from_file_location("eval_librispeech_backends", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_reference(manifest: Path) -> str:
    texts: list[str] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = str(item.get("text") or item.get("reference") or item.get("ref_text") or "")
            if not text:
                raise SystemExit(f"{manifest}:{line_no}: missing text/reference")
            texts.append(text)
    return " ".join(texts)


def rows_by_label(benchmark: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in benchmark.get("runs") or []:
        rows.setdefault(str(row.get("label")), []).append(row)
    return rows


def row_metric(row: dict[str, Any], flat_key: str, metrics_key: str | None = None) -> Any:
    if row.get(flat_key) is not None:
        return row.get(flat_key)
    metrics = row.get("metrics") or {}
    if isinstance(metrics, dict):
        return metrics.get(metrics_key or flat_key)
    return None


def score(benchmark_path: Path, manifest_path: Path) -> dict[str, Any]:
    helpers = load_eval_helpers()
    reference = read_reference(manifest_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

    labels: list[dict[str, Any]] = []
    for label, rows in rows_by_label(benchmark).items():
        runs = []
        for row in rows:
            edits, words, wer = helpers.wer_stats(reference, str(row.get("text") or ""))
            runs.append(
                {
                    "run_index": row.get("run_index"),
                    "edits": edits,
                    "words": words,
                    "wer": wer,
                    "real_audio_rtf": row_metric(
                        row,
                        "real_audio_rtf",
                        "real_audio_real_time_factor",
                    ),
                    "rtf": row_metric(row, "rtf", "real_time_factor"),
                    "decode_seconds": row_metric(row, "decode_seconds"),
                    "wall_seconds": row.get("wall_seconds"),
                }
            )
        numeric_medians = {}
        for key in ("real_audio_rtf", "rtf", "decode_seconds", "wall_seconds"):
            values = [float(run[key]) for run in runs if run.get(key) is not None]
            numeric_medians[f"median_{key}"] = statistics.median(values) if values else None
        labels.append(
            {
                "label": label,
                "runs": runs,
                "median_wer": statistics.median(run["wer"] for run in runs) if runs else None,
                "median_edits": statistics.median(run["edits"] for run in runs) if runs else None,
                "words": runs[0]["words"] if runs else 0,
                **numeric_medians,
            }
        )

    return {
        "benchmark": str(benchmark_path),
        "manifest": str(manifest_path),
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_json", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("build/librispeech-backend-eval/manifest.jsonl"),
        help="Manifest used to build the concatenated long WAV.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score(args.benchmark_json, args.manifest)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"benchmark: {result['benchmark']}")
    print(f"manifest: {result['manifest']}")
    for label in result["labels"]:
        wer_percent = 100.0 * float(label["median_wer"])
        perf_bits = []
        if label.get("median_real_audio_rtf") is not None:
            perf_bits.append(f"real-audio RTF={label['median_real_audio_rtf']:.3f}")
        if label.get("median_rtf") is not None:
            perf_bits.append(f"RTF={label['median_rtf']:.3f}")
        if label.get("median_decode_seconds") is not None:
            perf_bits.append(f"decode={label['median_decode_seconds']:.3f}s")
        perf_suffix = f" ({', '.join(perf_bits)})" if perf_bits else ""
        print(
            f"{label['label']}: median {label['median_edits']:.0f}/{label['words']} "
            f"WER={wer_percent:.2f}%{perf_suffix}"
        )
        for run in label["runs"]:
            run_perf = []
            if run.get("real_audio_rtf") is not None:
                run_perf.append(f"real-audio RTF={float(run['real_audio_rtf']):.3f}")
            if run.get("decode_seconds") is not None:
                run_perf.append(f"decode={float(run['decode_seconds']):.3f}s")
            run_suffix = f" ({', '.join(run_perf)})" if run_perf else ""
            print(
                f"  run {run['run_index']}: {run['edits']}/{run['words']} "
                f"WER={100.0 * run['wer']:.2f}%{run_suffix}"
            )


if __name__ == "__main__":
    main()
