from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, WordpipeConfig, load_config
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


def _cmd_model_info(args: argparse.Namespace) -> int:
    from .asr_worker import render_model_info

    print(render_model_info(Path(args.model_dir)))
    return 0


def _cmd_transcribe_file(args: argparse.Namespace) -> int:
    from .asr_worker import AsrWorkerConfig, transcribe_wav_file

    config = AsrWorkerConfig(
        model_dir=Path(args.model_dir),
        provider=args.provider,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
    )
    print(transcribe_wav_file(config, Path(args.wav)))
    return 0


def _cmd_download_model(args: argparse.Namespace) -> int:
    from .models import download_model, make_download_plan

    plan = make_download_plan(
        output_dir=Path(args.output_dir),
        repo_id=args.repo_id,
        include_test_wavs=args.include_test_wavs,
    )
    model_dir = download_model(plan, force=args.force)
    print(model_dir)
    return 0


def _cmd_type_text(args: argparse.Namespace) -> int:
    from .insertion import DryRunKeyboardBackend, PortalKeyboardBackend

    backend = DryRunKeyboardBackend() if args.dry_run else PortalKeyboardBackend()
    backend.open()
    try:
        backend.insert_text(args.text)
    finally:
        backend.close()

    if args.dry_run:
        for event in backend.events:
            print(event)
    return 0


def _load_cli_config(args: argparse.Namespace) -> WordpipeConfig:
    path = Path(args.config).expanduser() if getattr(args, "config", None) else None
    return load_config(path)


def _resolve_model_dir(args: argparse.Namespace, config: WordpipeConfig) -> Path:
    raw = getattr(args, "model_dir", None)
    model_dir = Path(raw).expanduser() if raw else config.model_dir
    if model_dir is None:
        raise SystemExit("--model-dir is required when config.toml does not set model_dir")
    return model_dir


def _cmd_daemon(args: argparse.Namespace) -> int:
    from .daemon import DaemonConfig, run_daemon
    from .transcript import make_transcript_sink

    file_config = _load_cli_config(args)
    config = DaemonConfig(
        model_dir=_resolve_model_dir(args, file_config),
        dry_run_insertion=args.dry_run_insertion or file_config.dry_run_insertion,
        provider=args.provider or file_config.provider,
        num_threads=args.num_threads if args.num_threads is not None else file_config.num_threads,
        sample_rate=args.sample_rate if args.sample_rate is not None else file_config.sample_rate,
        spoken_punctuation=file_config.spoken_punctuation and not args.no_spoken_punctuation,
    )
    return run_daemon(config, make_transcript_sink(args.overlay or file_config.overlay))


def _cmd_hotkey_daemon(args: argparse.Namespace) -> int:
    from .daemon import DaemonConfig, run_hotkey_daemon
    from .transcript import make_transcript_sink

    file_config = _load_cli_config(args)
    config = DaemonConfig(
        model_dir=_resolve_model_dir(args, file_config),
        dry_run_insertion=args.dry_run_insertion or file_config.dry_run_insertion,
        provider=args.provider or file_config.provider,
        num_threads=args.num_threads if args.num_threads is not None else file_config.num_threads,
        sample_rate=args.sample_rate if args.sample_rate is not None else file_config.sample_rate,
        spoken_punctuation=file_config.spoken_punctuation and not args.no_spoken_punctuation,
    )
    return run_hotkey_daemon(
        config,
        mode=args.mode or file_config.mode,
        shortcut=args.shortcut or file_config.shortcut,
        manual_hotkey=args.manual_hotkey,
        transcript=make_transcript_sink(args.overlay or file_config.overlay),
    )


def _cmd_config_example(_args: argparse.Namespace) -> int:
    print(DEFAULT_CONFIG, end="")
    return 0


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

    model_info = subparsers.add_parser(
        "model-info",
        help="Inspect a sherpa-onnx model directory and report the factory layout.",
    )
    model_info.add_argument(
        "--model-dir",
        required=True,
        help="Path to sherpa-onnx streaming model directory.",
    )
    model_info.set_defaults(func=_cmd_model_info)

    transcribe_file = subparsers.add_parser(
        "transcribe-file",
        help="Run streaming ASR over a 16 kHz mono PCM WAV file.",
    )
    transcribe_file.add_argument("--model-dir", required=True)
    transcribe_file.add_argument("--wav", required=True)
    transcribe_file.add_argument("--provider", default="cpu", help="ONNX Runtime provider.")
    transcribe_file.add_argument("--num-threads", type=int, default=2)
    transcribe_file.add_argument("--sample-rate", type=int, default=16000)
    transcribe_file.set_defaults(func=_cmd_transcribe_file)

    download_model = subparsers.add_parser(
        "download-model",
        help="Download the default Nemotron int8 streaming model from Hugging Face.",
    )
    download_model.add_argument(
        "--repo-id",
        default="csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        help="Hugging Face model repo ID.",
    )
    download_model.add_argument(
        "--output-dir",
        default="models",
        help="Directory that will contain the downloaded model directory.",
    )
    download_model.add_argument(
        "--include-test-wavs",
        action="store_true",
        help="Also download the small upstream test WAV files.",
    )
    download_model.add_argument("--force", action="store_true", help="Redownload existing files.")
    download_model.set_defaults(func=_cmd_download_model)

    type_text = subparsers.add_parser(
        "type-text",
        help="Insert text using the keyboard insertion backend.",
    )
    type_text.add_argument("text", help="Text to insert.")
    type_text.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated keysyms instead of opening a portal session.",
    )
    type_text.set_defaults(func=_cmd_type_text)

    config_example = subparsers.add_parser(
        "config-example",
        help="Print an example XDG config.toml.",
    )
    config_example.set_defaults(func=_cmd_config_example)

    daemon = subparsers.add_parser(
        "daemon",
        help="Run the MVP dictation loop: ASR subprocess plus text insertion.",
    )
    daemon.add_argument(
        "--model-dir",
        help="Path to sherpa-onnx Nemotron int8 streaming model directory.",
    )
    daemon.add_argument("--config", help="Path to config.toml.")
    daemon.add_argument(
        "--dry-run-insertion",
        action="store_true",
        help="Print keyboard events instead of opening a portal keyboard session.",
    )
    daemon.add_argument("--provider", help="ONNX Runtime provider.")
    daemon.add_argument("--num-threads", type=int)
    daemon.add_argument("--sample-rate", type=int)
    daemon.add_argument(
        "--no-spoken-punctuation",
        action="store_true",
        help="Insert raw ASR text instead of converting spoken punctuation commands.",
    )
    daemon.add_argument(
        "--overlay",
        choices=("stderr", "gtk"),
        help="Where partial transcript/status text is shown.",
    )
    daemon.set_defaults(func=_cmd_daemon)

    hotkey_daemon = subparsers.add_parser(
        "hotkey-daemon",
        help="Run dictation controlled by a GNOME GlobalShortcuts portal hotkey.",
    )
    hotkey_daemon.add_argument(
        "--model-dir",
        help="Path to sherpa-onnx Nemotron int8 streaming model directory.",
    )
    hotkey_daemon.add_argument("--config", help="Path to config.toml.")
    hotkey_daemon.add_argument(
        "--mode",
        choices=("hold", "toggle"),
        help="Shortcut behavior. Hold starts on activation and stops on deactivation.",
    )
    hotkey_daemon.add_argument(
        "--shortcut",
        help="Preferred GlobalShortcuts trigger string.",
    )
    hotkey_daemon.add_argument(
        "--manual-hotkey",
        action="store_true",
        help="Read manual commands from stdin instead of opening GlobalShortcuts.",
    )
    hotkey_daemon.add_argument(
        "--dry-run-insertion",
        action="store_true",
        help="Print keyboard events instead of opening a portal keyboard session.",
    )
    hotkey_daemon.add_argument("--provider", help="ONNX Runtime provider.")
    hotkey_daemon.add_argument("--num-threads", type=int)
    hotkey_daemon.add_argument("--sample-rate", type=int)
    hotkey_daemon.add_argument(
        "--no-spoken-punctuation",
        action="store_true",
        help="Insert raw ASR text instead of converting spoken punctuation commands.",
    )
    hotkey_daemon.add_argument(
        "--overlay",
        choices=("stderr", "gtk"),
        help="Where partial transcript/status text is shown.",
    )
    hotkey_daemon.set_defaults(func=_cmd_hotkey_daemon)

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
