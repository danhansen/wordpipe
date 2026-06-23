#!/usr/bin/env python3
"""Compare wordpipe-parakeet-worker --trace-token-decisions JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LANG_TAG_RE = re.compile(r"<[a-z]{2}(?:-[A-Z]{2})?>")


def load_trace(path: Path) -> tuple[list[dict[str, Any]], str]:
    decisions: list[dict[str, Any]] = []
    committed = ""
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if event.get("event") == "token_decision":
                decisions.append(dict(event["data"]))
            elif event.get("event") == "commit":
                committed = str(event.get("text") or "")
    return decisions, committed


def visible_piece(piece: str) -> str:
    return "" if LANG_TAG_RE.fullmatch(piece) else piece


def materialize(decisions: list[dict[str, Any]]) -> tuple[str, list[int]]:
    text = ""
    offsets: list[int] = []
    for decision in decisions:
        offsets.append(len(text))
        text += visible_piece(str(decision["piece"]))
    return text, offsets


def print_decision(label: str, decision: dict[str, Any]) -> None:
    keys = (
        "chunk_index",
        "frame_index",
        "symbol_index",
        "input_token_id",
        "token_id",
        "piece",
        "logit",
        "blank_logit",
        "margin",
    )
    summary = {key: decision[key] for key in keys}
    print(f"{label}: {summary}")
    for top in decision.get("top", [])[:8]:
        print(f"  {top['id']:>5} {top['piece']!r:<14} {top['logit']:.6f}")


def first_token_divergence(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> int | None:
    for index, (left_decision, right_decision) in enumerate(zip(left, right)):
        if left_decision["token_id"] != right_decision["token_id"]:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def token_index_for_offset(offsets: list[int], offset: int) -> int | None:
    if offset < 0:
        return None
    result = None
    for index, token_offset in enumerate(offsets):
        if token_offset > offset:
            break
        result = index
    return result


def print_window(
    title: str,
    decisions: list[dict[str, Any]],
    offsets: list[int],
    center: int,
    radius: int,
) -> None:
    print(f"\n{title}")
    for index in range(max(0, center - radius), min(len(decisions), center + radius + 1)):
        decision = decisions[index]
        mark = ">>" if index == center else "  "
        print(
            f"{mark} {index:04d} off={offsets[index]:04d} "
            f"c{decision['chunk_index']:03d} f{decision['frame_index']} "
            f"s{decision['symbol_index']} in={decision['input_token_id']} "
            f"id={decision['token_id']} piece={decision['piece']!r} "
            f"logit={decision['logit']:.3f} blank={decision['blank_logit']:.3f} "
            f"margin={decision['margin']:.3f}"
        )
        if abs(index - center) <= 3:
            tops = ", ".join(
                f"{top['id']}:{top['piece']!r}:{top['logit']:.3f}"
                for top in decision.get("top", [])[:8]
            )
            print(f"     top: {tops}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument(
        "--needle",
        action="append",
        default=[],
        help="Case-insensitive visible-text substring to locate in both traces. May be repeated.",
    )
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    left, left_text = load_trace(args.left)
    right, right_text = load_trace(args.right)
    left_visible, left_offsets = materialize(left)
    right_visible, right_offsets = materialize(right)

    print(f"{args.left_label}: decisions={len(left)} committed_len={len(left_text)}")
    print(f"{args.right_label}: decisions={len(right)} committed_len={len(right_text)}")
    divergence = first_token_divergence(left, right)
    if divergence is None:
        print("first token divergence: none")
    else:
        print(f"first token divergence: {divergence}")
        if divergence < len(left):
            print_decision(args.left_label, left[divergence])
        if divergence < len(right):
            print_decision(args.right_label, right[divergence])

    for needle in args.needle:
        print(f"\nneedle: {needle!r}")
        for label, decisions, visible, offsets in (
            (args.left_label, left, left_visible, left_offsets),
            (args.right_label, right, right_visible, right_offsets),
        ):
            offset = visible.lower().find(needle.lower())
            print(f"{label}: offset={offset}")
            if offset < 0:
                continue
            start = max(0, offset - 70)
            end = min(len(visible), offset + len(needle) + 90)
            print(visible[start:end])
            index = token_index_for_offset(offsets, offset)
            if index is not None:
                print_window(label, decisions, offsets, index, args.window)


if __name__ == "__main__":
    main()
