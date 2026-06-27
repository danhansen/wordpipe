#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wordpipe.models import default_model_root, profile_runtime_dir  # noqa: E402


DEFAULT_SMOKE_WAVS = (
    REPO_ROOT / "build/librispeech-backend-eval-smoke/wavs/1089-134686-0019.wav",
    REPO_ROOT / "models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11/test_wavs/en.wav",
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wav = resolve_wav(args.wav)
    model_dir = resolve_model_dir(args)
    command = build_command(args, model_dir, wav)

    if args.print_command:
        print(" ".join(command))
        return 0

    process = subprocess.run(command, text=True, capture_output=True)
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    if process.returncode != 0:
        if process.stdout:
            print(process.stdout, end="")
        return process.returncode

    summary = summarize_events(process.stdout)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["commit_text"]:
        print("wordpipe smoke failed: no commit text produced", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test a Wordpipe Parakeet/Nemotron model with stream-file-test."
    )
    parser.add_argument("--model-dir", type=Path, help="Runtime model directory to test.")
    parser.add_argument(
        "--model-profile",
        choices=("fast", "compact"),
        default="compact",
        help="Model profile to resolve when --model-dir is not provided.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        help="Directory containing model profiles. Defaults to the local Wordpipe model root.",
    )
    parser.add_argument("--wav", type=Path, help="16 kHz mono WAV to feed through ASR.")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--flush-chunks", type=int, default=3)
    parser.add_argument(
        "--command",
        default=str(REPO_ROOT / "scripts/wordpipe-dev"),
        help="Local wordpipe command to run.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved command without running it.",
    )
    return parser.parse_args(argv)


def resolve_wav(wav: Path | None) -> Path:
    if wav is not None:
        return wav.expanduser().resolve()
    for candidate in DEFAULT_SMOKE_WAVS:
        if candidate.exists():
            return candidate.resolve()
    raise SystemExit("missing --wav and no default smoke WAV exists under build/ or models/")


def resolve_model_dir(args: argparse.Namespace) -> Path:
    if args.model_dir is not None:
        return args.model_dir.expanduser().resolve()
    model_root = args.model_root.expanduser() if args.model_root else default_model_root()
    return profile_runtime_dir(model_root, args.model_profile).resolve()


def build_command(args: argparse.Namespace, model_dir: Path, wav: Path) -> list[str]:
    command = [args.command]

    command.extend(
        [
            "stream-file-test",
            "--asr-runtime",
            "parakeet",
            "--model-dir",
            str(model_dir),
            "--wav",
            str(wav),
            "--num-threads",
            str(args.num_threads),
            "--flush-chunks",
            str(args.flush_chunks),
            "--json",
        ]
    )
    return command


def summarize_events(output: str) -> dict[str, Any]:
    events = []
    for line in output.splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))

    commits = [event for event in events if event.get("event") == "commit"]
    partials = [event for event in events if event.get("event") == "partial"]
    stats = [event for event in events if event.get("event") == "stats"]
    last_stats = stats[-1].get("data", {}) if stats else {}
    return {
        "events": len(events),
        "partials": len(partials),
        "commits": len(commits),
        "commit_text": commits[-1].get("text", "") if commits else "",
        "real_time_factor": last_stats.get("real_time_factor"),
        "audio_seconds": last_stats.get("audio_seconds"),
        "decode_seconds": last_stats.get("decode_seconds"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
