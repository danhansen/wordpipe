from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock


def _load_download_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_hf_ranged.py"
    spec = importlib.util.spec_from_file_location("download_hf_ranged", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    status = 206

    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def read(self, size: int) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class RangedDownloadTests(unittest.TestCase):
    def test_oversized_part_is_discarded_and_redownloaded(self) -> None:
        module = _load_download_module()
        payload = b"abcde"

        with tempfile.TemporaryDirectory() as tmp:
            part_path = Path(tmp) / "model.onnx.data.part00"
            part_path.write_bytes(b"stale-corrupt-extra")
            progress = [len(payload)]

            def urlopen(request, timeout):  # type: ignore[no-untyped-def]
                self.assertEqual(timeout, 60)
                self.assertEqual(request.get_header("Range"), "bytes=10-14")
                return FakeResponse(payload)

            with mock.patch.object(module.urllib.request, "urlopen", side_effect=urlopen):
                module.download_range(
                    "https://example.test/model.onnx.data",
                    part_path,
                    10,
                    14,
                    2,
                    progress,
                    0,
                    threading.Lock(),
                    0,
                )

            self.assertEqual(part_path.read_bytes(), payload)
            self.assertEqual(progress[0], len(payload))


if __name__ == "__main__":
    unittest.main()
