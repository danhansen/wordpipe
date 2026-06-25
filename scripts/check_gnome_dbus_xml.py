#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUST_PROTOCOL = ROOT / "crates/wordpipe-protocol/src/lib.rs"
GNOME_CLIENTS = [
    ROOT / "extensions/gnome-shell/wordpipe@dhansen.dev/extension.js",
    ROOT / "extensions/gnome-shell/wordpipe@dhansen.dev/prefs.js",
]


def main() -> int:
    expected = interface_shape(extract_rust_xml(RUST_PROTOCOL))
    errors: list[str] = []
    for path in GNOME_CLIENTS:
        actual = interface_shape(extract_js_xml(path))
        if actual != expected:
            errors.append(format_difference(path, expected, actual))
    if errors:
        print("\n\n".join(errors), file=sys.stderr)
        return 1
    return 0


def extract_rust_xml(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'pub const INTROSPECTION_XML: &str = r#"\n(?P<xml>.*?)\n"#;',
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"could not find INTROSPECTION_XML in {path}")
    return match.group("xml")


def extract_js_xml(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"const SERVICE_XML = `\n(?P<xml>.*?)`;",
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"could not find SERVICE_XML in {path}")
    return match.group("xml")


Signature = tuple[str, tuple[tuple[str, str, str], ...]]


def interface_shape(xml: str) -> dict[str, tuple[Signature, ...]]:
    root = ET.fromstring(xml)
    interface = root.find("interface")
    if interface is None:
        raise RuntimeError("D-Bus XML does not contain an interface")
    return {
        "methods": tuple(signature(child) for child in interface.findall("method")),
        "signals": tuple(signature(child) for child in interface.findall("signal")),
    }


def signature(element: ET.Element) -> Signature:
    return (
        element.attrib["name"],
        tuple(
            (
                arg.attrib.get("name", ""),
                arg.attrib["type"],
                arg.attrib.get("direction", ""),
            )
            for arg in element.findall("arg")
        ),
    )


def format_difference(
    path: Path,
    expected: dict[str, tuple[Signature, ...]],
    actual: dict[str, tuple[Signature, ...]],
) -> str:
    lines = [f"{path.relative_to(ROOT)} D-Bus XML differs from Rust protocol"]
    for key in ("methods", "signals"):
        expected_by_name = {item[0]: item for item in expected[key]}
        actual_by_name = {item[0]: item for item in actual[key]}
        missing = sorted(set(expected_by_name) - set(actual_by_name))
        extra = sorted(set(actual_by_name) - set(expected_by_name))
        if missing:
            lines.append(f"  missing {key}: {', '.join(missing)}")
        if extra:
            lines.append(f"  extra {key}: {', '.join(extra)}")
        for name in sorted(set(expected_by_name) & set(actual_by_name)):
            if expected_by_name[name] != actual_by_name[name]:
                lines.append(f"  {key} signature differs for {name}")
        if tuple(item[0] for item in expected[key]) != tuple(item[0] for item in actual[key]):
            lines.append(f"  {key} order differs")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
