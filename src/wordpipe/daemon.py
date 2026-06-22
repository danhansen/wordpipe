from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterator

from .insertion import DryRunKeyboardBackend, KeyboardBackend, PortalKeyboardBackend


@dataclass(frozen=True)
class DaemonConfig:
    model_dir: Path
    dry_run_insertion: bool = False
    provider: str = "cpu"
    num_threads: int = 2
    sample_rate: int = 16000


def format_committed_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.endswith((".", ",", "!", "?", ";", ":", "\n")):
        return f"{stripped} "
    return f"{stripped} "


class AsrProcess:
    def __init__(self, config: DaemonConfig) -> None:
        self._config = config
        self._proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._proc is not None:
            return

        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1])
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = source_root if not existing else f"{source_root}{os.pathsep}{existing}"

        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "wordpipe",
                "asr-worker",
                "--model-dir",
                str(self._config.model_dir),
                "--provider",
                self._config.provider,
                "--num-threads",
                str(self._config.num_threads),
                "--sample-rate",
                str(self._config.sample_rate),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    def send(self, command: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("ASR process is not running")
        self._proc.stdin.write(json.dumps({"command": command}) + "\n")
        self._proc.stdin.flush()

    def events(self) -> Iterator[dict[str, object]]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("ASR process is not running")
        for line in self._proc.stdout:
            if not line.strip():
                continue
            yield json.loads(line)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self.send("shutdown")
                self._proc.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        finally:
            self._proc = None


class DictationDaemon:
    def __init__(self, config: DaemonConfig, keyboard: KeyboardBackend) -> None:
        self._config = config
        self._keyboard = keyboard
        self._asr = AsrProcess(config)

    def run(self) -> int:
        self._keyboard.open()
        if isinstance(self._keyboard, DryRunKeyboardBackend):
            self._keyboard.events.clear()
        self._asr.start()
        try:
            self._asr.send("start")
            for item in self._asr.events():
                kind = item.get("event")
                if kind == "ready":
                    print("wordpipe: ASR worker ready", file=sys.stderr)
                elif kind == "listening":
                    print("wordpipe: listening", file=sys.stderr)
                elif kind == "partial":
                    print(f"partial: {item.get('text', '')}", file=sys.stderr)
                elif kind == "commit":
                    text = format_committed_text(str(item.get("text", "")))
                    if text:
                        print(f"commit: {text}", file=sys.stderr)
                        self._keyboard.insert_text(text)
                        if isinstance(self._keyboard, DryRunKeyboardBackend):
                            for rendered in self._keyboard.events:
                                print(f"key: {rendered}", file=sys.stderr)
                            self._keyboard.events.clear()
                elif kind == "error":
                    print(f"wordpipe error: {item.get('message', '')}", file=sys.stderr)
                    return 1
                elif kind == "stopped":
                    return 0
        finally:
            self._asr.close()
            self._keyboard.close()
        return 0


def run_daemon(config: DaemonConfig) -> int:
    keyboard: KeyboardBackend
    keyboard = DryRunKeyboardBackend() if config.dry_run_insertion else PortalKeyboardBackend()
    daemon = DictationDaemon(config, keyboard)
    return daemon.run()
