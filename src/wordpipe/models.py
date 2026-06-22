from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from urllib.request import urlretrieve


DEFAULT_MODEL_REPO = (
    "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
)
DEFAULT_MODEL_FILES = (
    "tokens.txt",
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "README.md",
)


@dataclass(frozen=True)
class DownloadPlan:
    repo_id: str
    output_dir: Path
    files: tuple[str, ...]

    @property
    def model_dir(self) -> Path:
        return self.output_dir / self.repo_id.split("/")[-1]


def make_download_plan(
    output_dir: Path,
    repo_id: str = DEFAULT_MODEL_REPO,
    include_test_wavs: bool = False,
) -> DownloadPlan:
    files = list(DEFAULT_MODEL_FILES)
    if include_test_wavs:
        files.extend(["test_wavs/en.wav", "test_wavs/ja.wav"])
    return DownloadPlan(repo_id=repo_id, output_dir=output_dir.expanduser(), files=tuple(files))


def download_model(plan: DownloadPlan, force: bool = False) -> Path:
    model_dir = plan.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in plan.files:
        destination = model_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            print(f"skip {destination}", file=sys.stderr)
            continue
        url = model_file_url(plan.repo_id, filename)
        print(f"download {url}", file=sys.stderr)
        temporary = destination.with_name(destination.name + ".part")
        urlretrieve(url, temporary, _progress_reporter(destination))
        temporary.replace(destination)
    return model_dir


def model_file_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _progress_reporter(destination: Path):
    last_report = 0.0

    def report(block_count: int, block_size: int, total_size: int) -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report < 5 and block_count != 0:
            return
        last_report = now
        downloaded = block_count * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            print(
                f"  {destination.name}: {downloaded}/{total_size} bytes ({percent:.1f}%)",
                file=sys.stderr,
            )
        else:
            print(f"  {destination.name}: {downloaded} bytes", file=sys.stderr)

    return report
