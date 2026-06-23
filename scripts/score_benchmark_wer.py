#!/usr/bin/env python3
"""Score rough WER for benchmark_parakeet_variant.py output.

The long-WAV benchmark concatenates a small LibriSpeech manifest into one WAV
and stores the final transcript for each run. This script reconstructs that
concatenated reference and applies the same normalization/edit-distance helpers
used by eval_librispeech_backends.py.
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
                }
            )
        labels.append(
            {
                "label": label,
                "runs": runs,
                "median_wer": statistics.median(run["wer"] for run in runs) if runs else None,
                "median_edits": statistics.median(run["edits"] for run in runs) if runs else None,
                "words": runs[0]["words"] if runs else 0,
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
        print(
            f"{label['label']}: median {label['median_edits']:.0f}/{label['words']} "
            f"WER={wer_percent:.2f}%"
        )
        for run in label["runs"]:
            print(
                f"  run {run['run_index']}: {run['edits']}/{run['words']} "
                f"WER={100.0 * run['wer']:.2f}%"
            )


if __name__ == "__main__":
    main()
