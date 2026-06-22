#!/usr/bin/env python3
"""Compare Wordpipe's Parakeet and sherpa backends on a LibriSpeech sample."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import string
import subprocess
import sys
import tarfile
import time
from typing import Any


SAMPLE_RATE = 16000
DEFAULT_LIBRISPEECH_TAR = Path("~/Downloads/test-clean.tar.gz").expanduser()
DEFAULT_SHERPA_MODEL = Path(
    "models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
)
DEFAULT_PARAKEET_MODEL = Path(
    "models/nemotron-3.5-asr-streaming-0.6b-parakeet-int8-projected-c56"
)


@dataclass
class Sample:
    utt_id: str
    audio: Path
    text: str
    duration_sec: float


def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def safe_extract_tar(tar_path: Path, dest_dir: Path) -> Path:
    marker = dest_dir / ".complete"
    if marker.exists():
        candidates = sorted(
            path
            for path in dest_dir.rglob("test-clean")
            if path.is_dir() and any(path.rglob("*.trans.txt"))
        )
        if candidates:
            return candidates[0]

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_real = dest_dir.resolve()
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive.getmembers():
            target = (dest_dir / member.name).resolve()
            if target != dest_real and dest_real not in target.parents:
                raise RuntimeError(f"Refusing unsafe tar member: {member.name}")
        archive.extractall(dest_dir)

    marker.write_text("ok\n", encoding="utf-8")
    candidates = sorted(
        path
        for path in dest_dir.rglob("test-clean")
        if path.is_dir() and any(path.rglob("*.trans.txt"))
    )
    if not candidates:
        raise RuntimeError(f"No LibriSpeech test-clean directory found after extracting {tar_path}")
    return candidates[0]


def ffprobe_duration(path: Path) -> float:
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(proc.stdout.strip())


def parse_librispeech_root(root: Path, *, candidate_count: int | None) -> list[Sample]:
    transcripts: dict[str, str] = {}
    for transcript in sorted(root.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            utt_id, text = line.split(" ", 1)
            transcripts[utt_id] = text

    samples: list[Sample] = []
    audio_paths = sorted(root.rglob("*.flac"))
    if candidate_count is not None:
        audio_paths = audio_paths[:candidate_count]
    for audio_path in audio_paths:
        text = transcripts.get(audio_path.stem)
        if text is None:
            continue
        samples.append(Sample(audio_path.stem, audio_path, text, ffprobe_duration(audio_path)))
    if not samples:
        raise RuntimeError(f"No LibriSpeech samples found under {root}")
    return samples


def parse_manifest(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        audio = Path(item["audio"]).expanduser()
        text = str(item.get("text") or item.get("reference") or item.get("ref_text") or "")
        if not text:
            raise ValueError(f"{path}:{line_no}: missing text/reference")
        duration = float(item.get("duration_sec") or ffprobe_duration(audio))
        samples.append(Sample(str(item.get("utt_id") or audio.stem), audio, text, duration))
    if not samples:
        raise RuntimeError(f"No samples found in {path}")
    return samples


def select_samples(samples: list[Sample], *, count: int, max_audio_sec: float | None, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    selected: list[Sample] = []
    total = 0.0
    for sample in shuffled:
        if len(selected) >= count:
            break
        if max_audio_sec is not None and selected and total + sample.duration_sec > max_audio_sec:
            continue
        selected.append(sample)
        total += sample.duration_sec
        if max_audio_sec is not None and total >= max_audio_sec:
            break
    selected.sort(key=lambda sample: sample.utt_id)
    return selected


def wav_for_sample(sample: Sample, wav_dir: Path) -> Path:
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


def normalize_words(text: str) -> list[str]:
    table = str.maketrans({ch: " " for ch in string.punctuation})
    normalized = text.lower().replace("’", "'").translate(table)
    return normalized.split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ref_word != hyp_word),
            )
        prev = curr
    return prev[-1]


def wer_stats(ref_text: str, hyp_text: str) -> tuple[int, int, float]:
    ref = normalize_words(ref_text)
    hyp = normalize_words(hyp_text)
    edits = edit_distance(ref, hyp)
    words = len(ref)
    return edits, words, edits / max(words, 1)


def parse_json_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        events.append(json.loads(line))
    return events


def final_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    commits = [event for event in events if event.get("event") == "commit"]
    if commits:
        return commits[-1]
    text_events = [event for event in events if event.get("text")]
    if text_events:
        return text_events[-1]
    return events[-1] if events else {"text": "", "data": {}}


def default_ort_dylib() -> Path | None:
    for pattern in (
        ".venv-nemo-export/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*",
        ".venv/lib/python*/site-packages/sherpa_onnx/lib/libonnxruntime.so*",
    ):
        matches = sorted(Path(".").glob(pattern))
        if matches:
            return matches[0].resolve()
    return None


def run_parakeet(args: argparse.Namespace, wav: Path) -> tuple[str, dict[str, Any], str]:
    env = os.environ.copy()
    if "ORT_DYLIB_PATH" not in env:
        dylib = default_ort_dylib()
        if dylib is not None:
            env["ORT_DYLIB_PATH"] = str(dylib)
    env.setdefault("OMP_NUM_THREADS", str(args.num_threads))
    env.setdefault("MKL_NUM_THREADS", str(args.num_threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(args.num_threads))
    command = [
        str(args.parakeet_worker),
        "--model-dir",
        str(args.parakeet_model_dir),
        "--wav",
        str(wav),
        "--num-threads",
        str(args.num_threads),
        "--graph-optimization",
        args.graph_optimization,
        "--flush-chunks",
        str(args.flush_chunks),
    ]
    proc = run(command, env=env, timeout=args.timeout_seconds)
    event = final_event(parse_json_events(proc.stdout))
    return str(event.get("text") or ""), dict(event.get("data") or {}), proc.stderr


def run_sherpa(args: argparse.Namespace, wav: Path) -> tuple[str, dict[str, Any], str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    command = [
        sys.executable,
        "-m",
        "wordpipe",
        "stream-file-test",
        "--asr-runtime",
        "sherpa",
        "--model-dir",
        str(args.sherpa_model_dir),
        "--wav",
        str(wav),
        "--num-threads",
        str(args.num_threads),
        "--chunk-seconds",
        str(args.chunk_seconds),
        "--flush-chunks",
        str(args.flush_chunks),
        "--json",
    ]
    proc = run(command, env=env, timeout=args.timeout_seconds)
    event = final_event(parse_json_events(proc.stdout))
    return str(event.get("text") or ""), dict(event.get("data") or {}), proc.stderr


def evaluate_backend(
    name: str,
    samples: list[Sample],
    args: argparse.Namespace,
    wav_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    runner = run_parakeet if name == "parakeet" else run_sherpa
    total_edits = 0
    total_words = 0
    total_audio = 0.0
    total_processed = 0.0
    total_decode = 0.0
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as out:
        for idx, sample in enumerate(samples, start=1):
            wav = wav_for_sample(sample, wav_dir)
            item_started = time.perf_counter()
            hyp, metrics, stderr = runner(args, wav)
            elapsed = time.perf_counter() - item_started
            edits, words, item_wer = wer_stats(sample.text, hyp)
            total_edits += edits
            total_words += words
            total_audio += float(metrics.get("audio_seconds") or sample.duration_sec)
            total_processed += float(metrics.get("processed_audio_seconds") or sample.duration_sec)
            total_decode += float(metrics.get("decode_seconds") or 0.0)
            row = {
                "backend": name,
                "idx": idx,
                "utt_id": sample.utt_id,
                "audio": str(sample.audio),
                "wav": str(wav),
                "duration_sec": sample.duration_sec,
                "elapsed_sec": elapsed,
                "reference": sample.text,
                "hypothesis": hyp,
                "wer": item_wer,
                "edits": edits,
                "words": words,
                "metrics": metrics,
                "stderr_tail": stderr[-1000:] if stderr else "",
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"[eval] {name} {idx}/{len(samples)} "
                f"wer={item_wer:.3f} rtf={metrics.get('real_time_factor')} "
                f"hyp={hyp[:70]!r}",
                flush=True,
            )
    elapsed_total = time.perf_counter() - started
    return {
        "backend": name,
        "samples": len(samples),
        "audio_seconds": total_audio,
        "processed_audio_seconds": total_processed,
        "decode_seconds": total_decode,
        "elapsed_seconds": elapsed_total,
        "edits": total_edits,
        "words": total_words,
        "wer": total_edits / max(total_words, 1),
        "decode_rtf": total_decode / total_processed if total_processed else None,
        "real_audio_decode_rtf": total_decode / total_audio if total_audio else None,
        "wall_rtf": elapsed_total / total_audio if total_audio else None,
        "results": str(output_path),
    }


def write_manifest(samples: list[Sample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as out:
        for sample in samples:
            out.write(
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-tar", type=Path, default=DEFAULT_LIBRISPEECH_TAR)
    parser.add_argument("--librispeech-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("build/librispeech-backend-eval"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--candidate-count", type=int, default=120)
    parser.add_argument("--max-audio-sec", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=0.56)
    parser.add_argument("--flush-chunks", type=int, default=3)
    parser.add_argument("--graph-optimization", default="all")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--parakeet-worker", type=Path, default=Path("target/release/wordpipe-parakeet-worker"))
    parser.add_argument("--parakeet-model-dir", type=Path, default=DEFAULT_PARAKEET_MODEL)
    parser.add_argument("--sherpa-model-dir", type=Path, default=DEFAULT_SHERPA_MODEL)
    parser.add_argument("--backend", choices=("both", "parakeet", "sherpa"), default="both")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        samples = parse_manifest(args.manifest)
    elif args.librispeech_root:
        samples = parse_librispeech_root(args.librispeech_root, candidate_count=args.candidate_count)
    else:
        if not args.librispeech_tar.exists():
            raise SystemExit(f"Missing LibriSpeech tarball: {args.librispeech_tar}")
        root = safe_extract_tar(args.librispeech_tar, args.work_dir / "extracted")
        samples = parse_librispeech_root(root, candidate_count=args.candidate_count)

    subset = select_samples(
        samples,
        count=args.count,
        max_audio_sec=args.max_audio_sec,
        seed=args.seed,
    )
    if not subset:
        raise SystemExit("No samples selected.")
    manifest_path = args.work_dir / "manifest.jsonl"
    write_manifest(subset, manifest_path)
    print(
        f"[dataset] selected {len(subset)} utterances, "
        f"{sum(sample.duration_sec for sample in subset):.1f}s audio -> {manifest_path}",
        flush=True,
    )

    if args.backend in ("both", "parakeet") and not args.parakeet_worker.exists():
        raise SystemExit(f"Missing parakeet worker: {args.parakeet_worker}")

    backends = ["parakeet", "sherpa"] if args.backend == "both" else [args.backend]
    summaries = []
    for backend in backends:
        summaries.append(
            evaluate_backend(
                backend,
                subset,
                args,
                args.work_dir / "wavs",
                args.work_dir / f"{backend}.jsonl",
            )
        )

    summary = {
        "manifest": str(manifest_path),
        "settings": {
            "count": len(subset),
            "num_threads": args.num_threads,
            "chunk_seconds": args.chunk_seconds,
            "flush_chunks": args.flush_chunks,
            "graph_optimization": args.graph_optimization,
        },
        "backends": summaries,
    }
    summary_path = args.work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[summary] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
