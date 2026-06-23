#!/usr/bin/env python3
"""Build a concatenated LibriSpeech WAV and matching manifest for WER sweeps."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SAMPLE_RATE = 16000


def load_eval_helpers() -> Any:
    path = Path(__file__).resolve().parent / "eval_librispeech_backends.py"
    spec = importlib.util.spec_from_file_location("eval_librispeech_backends", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def convert_to_wav(sample: Any, wav_dir: Path) -> Path:
    wav_dir.mkdir(parents=True, exist_ok=True)
    output = wav_dir / f"{sample.utt_id}.wav"
    if output.exists():
        return output
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(sample.audio),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            str(output),
        ]
    )
    return output


def write_concat_list(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for path in paths:
            escaped = str(path.resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")


def write_manifest(samples: list[Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    {
                        "utt_id": sample.utt_id,
                        "audio": str(sample.audio),
                        "text": sample.text,
                        "duration_sec": sample.duration_sec,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        default=Path("build/librispeech-backend-eval/extracted/LibriSpeech/test-clean"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("build/librispeech-long-wav"))
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--candidate-count", type=int, default=800)
    parser.add_argument("--max-audio-sec", type=float, default=420.0)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--wav", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    helpers = load_eval_helpers()
    samples = helpers.parse_librispeech_root(
        args.librispeech_root,
        candidate_count=args.candidate_count,
    )
    selected = helpers.select_samples(
        samples,
        count=args.count,
        max_audio_sec=args.max_audio_sec,
        seed=args.seed,
    )
    if not selected:
        raise SystemExit("No samples selected")

    manifest = args.manifest or args.work_dir / "manifest.jsonl"
    wav = args.wav or args.work_dir / "librispeech-long.wav"
    wav_dir = args.work_dir / "wavs"
    concat_list = args.work_dir / "concat.txt"

    wav_paths = [convert_to_wav(sample, wav_dir) for sample in selected]
    write_manifest(selected, manifest)
    write_concat_list(wav_paths, concat_list)
    wav.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(wav),
        ]
    )

    total_sec = sum(float(sample.duration_sec) for sample in selected)
    total_words = sum(len(helpers.normalize_words(sample.text)) for sample in selected)
    summary = {
        "manifest": str(manifest),
        "wav": str(wav),
        "samples": len(selected),
        "audio_seconds": total_sec,
        "words": total_words,
        "seed": args.seed,
        "count": args.count,
        "candidate_count": args.candidate_count,
        "max_audio_sec": args.max_audio_sec,
    }
    (args.work_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
