from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .probe import ProbeResult, run_probe


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _cmd_probe(args: argparse.Namespace) -> int:
    result = run_probe()
    if args.json:
        _print_json(result.to_dict())
        return 0 if result.usable else 2

    print(result.render_text())
    return 0 if result.usable else 2


def _cmd_asr_worker(args: argparse.Namespace) -> int:
    from .asr_worker import AsrWorkerConfig, run_stdio_worker

    config = AsrWorkerConfig(
        model_dir=Path(args.model_dir),
        provider=args.provider,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
    )
    return run_stdio_worker(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wordpipe",
        description="Wayland-first GNOME dictation with streaming sherpa-onnx ASR.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser(
        "probe",
        help="Check GNOME, portal, and Python runtime capabilities.",
    )
    probe.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    probe.set_defaults(func=_cmd_probe)

    asr = subparsers.add_parser(
        "asr-worker",
        help="Run the streaming ASR worker protocol on stdin/stdout.",
    )
    asr.add_argument(
        "--model-dir",
        required=True,
        help="Path to sherpa-onnx Nemotron int8 streaming model directory.",
    )
    asr.add_argument("--provider", default="cpu", help="ONNX Runtime provider.")
    asr.add_argument("--num-threads", type=int, default=2)
    asr.add_argument("--sample-rate", type=int, default=16000)
    asr.set_defaults(func=_cmd_asr_worker)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


__all__ = ["ProbeResult", "build_parser", "main"]
