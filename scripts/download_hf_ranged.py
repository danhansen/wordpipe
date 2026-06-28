#!/usr/bin/env python3
"""Download a large Hugging Face file with resumable ranged requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--part-dir", type=pathlib.Path)
    parser.add_argument("--chunk-size", type=int, default=1024 * 256)
    parser.add_argument("--max-retries", type=int, default=200)
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args()


def request_headers(start: int | None = None, end: int | None = None) -> dict[str, str]:
    headers = {"User-Agent": "wordpipe-hf-ranged-downloader/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
    return headers


def ranges_for_size(size: int, workers: int) -> list[tuple[int, int, int]]:
    step = (size + workers - 1) // workers
    ranges: list[tuple[int, int, int]] = []
    for index, start in enumerate(range(0, size, step)):
        end = min(size - 1, start + step - 1)
        ranges.append((index, start, end))
    return ranges


def download_range(
    url: str,
    part_path: pathlib.Path,
    start: int,
    end: int,
    chunk_size: int,
    progress: list[int],
    index: int,
    lock: threading.Lock,
    max_retries: int,
) -> None:
    expected = end - start + 1
    for attempt in range(max_retries + 1):
        existing = part_path.stat().st_size if part_path.exists() else 0
        if existing == expected:
            return
        if existing > expected:
            raise RuntimeError(f"{part_path} is larger than its assigned range")

        resume_start = start + existing
        req = urllib.request.Request(url, headers=request_headers(resume_start, end))
        mode = "ab" if existing else "wb"
        with urllib.request.urlopen(req, timeout=60) as response:
            status = getattr(response, "status", None)
            if status != 206:
                raise RuntimeError(f"unexpected HTTP status {status} for range {resume_start}-{end}")
            with part_path.open(mode) as f:
                while True:
                    block = response.read(chunk_size)
                    if not block:
                        break
                    f.write(block)
                    with lock:
                        progress[index] += len(block)
        if attempt < max_retries:
            time.sleep(min(2.0, 0.05 * (attempt + 1)))

    actual = part_path.stat().st_size if part_path.exists() else 0
    raise RuntimeError(f"{part_path} has {actual} bytes, expected {expected}")


def progress_reporter(total: int, progress: list[int], lock: threading.Lock, stop: threading.Event) -> None:
    last_done = 0
    last_time = time.monotonic()
    while not stop.wait(5):
        with lock:
            done = sum(progress)
        now = time.monotonic()
        rate = (done - last_done) / max(now - last_time, 0.001)
        pct = done / total * 100.0
        progress_json = progress_payload(done, total, rate)
        if progress_json:
            print("wordpipe-progress " + json.dumps(progress_json, sort_keys=True), flush=True)
        else:
            print(
                f"{done / 1024 / 1024:.1f} MiB / {total / 1024 / 1024:.1f} MiB ({pct:.1f}%), "
                f"{rate / 1024:.1f} KiB/s",
                flush=True,
            )
        last_done = done
        last_time = now


def progress_payload(done: int, total: int, rate: float) -> dict[str, object] | None:
    profile = os.environ.get("WORDPIPE_PROGRESS_PROFILE")
    filename = os.environ.get("WORDPIPE_PROGRESS_FILENAME")
    if not profile or not filename:
        return None
    completed_base = int(os.environ.get("WORDPIPE_PROGRESS_COMPLETED_BASE", "0"))
    total_bytes = int(os.environ.get("WORDPIPE_PROGRESS_TOTAL_BYTES", str(total)))
    file_index = int(os.environ.get("WORDPIPE_PROGRESS_FILE_INDEX", "1"))
    file_count = int(os.environ.get("WORDPIPE_PROGRESS_FILE_COUNT", "1"))
    completed_bytes = completed_base + done
    fraction = completed_bytes / total_bytes if total_bytes > 0 else 0.0
    return {
        "profile": profile,
        "phase": "downloading",
        "message": (
            f"Downloading {filename} "
            f"({done / 1024 / 1024:.1f} MiB / {total / 1024 / 1024:.1f} MiB, "
            f"{rate / 1024:.1f} KiB/s)"
        ),
        "filename": filename,
        "file_index": file_index,
        "file_count": file_count,
        "file_size": total,
        "file_completed_bytes": done,
        "completed_bytes": completed_bytes,
        "total_bytes": total_bytes,
        "bytes_per_second": rate,
        "fraction": max(0.0, min(1.0, fraction)),
    }


def assemble(output: pathlib.Path, part_paths: list[pathlib.Path], expected_size: int) -> None:
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as out:
        for path in part_paths:
            with path.open("rb") as part:
                while True:
                    block = part.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
    actual = tmp.stat().st_size
    if actual != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"assembled file has {actual} bytes, expected {expected_size}")
    tmp.replace(output)


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    part_dir = args.part_dir or args.output.with_name(args.output.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)

    ranges = ranges_for_size(args.size, args.workers)
    part_paths = [part_dir / f"{args.output.name}.part{index:02d}" for index, _, _ in ranges]
    progress = []
    for path, (_, start, end) in zip(part_paths, ranges):
        expected = end - start + 1
        existing = path.stat().st_size if path.exists() else 0
        progress.append(min(existing, expected))
    lock = threading.Lock()
    stop = threading.Event()
    reporter = threading.Thread(target=progress_reporter, args=(args.size, progress, lock, stop), daemon=True)
    reporter.start()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    download_range,
                    args.url,
                    part_paths[index],
                    start,
                    end,
                    args.chunk_size,
                    progress,
                    index,
                    lock,
                    args.max_retries,
                )
                for index, start, end in ranges
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        stop.set()
        print(f"download failed: {exc}", file=sys.stderr)
        return 1

    stop.set()
    assemble(args.output, part_paths, args.size)
    if not args.keep_parts:
        for path in part_paths:
            path.unlink(missing_ok=True)
        try:
            part_dir.rmdir()
        except OSError:
            pass
    print(f"downloaded {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
