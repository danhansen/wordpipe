from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_flatpak_manifest_and_desktop_entry_use_launcher_wrapper(self) -> None:
        manifest = (ROOT / "packaging/flatpak/dev.wordpipe.Wordpipe.yml").read_text(
            encoding="utf-8"
        )
        desktop = (
            ROOT / "packaging/applications/dev.wordpipe.Wordpipe.desktop"
        ).read_text(encoding="utf-8")

        self.assertIn("command: wordpipe-flatpak-launch", manifest)
        self.assertIn("Exec=wordpipe-flatpak-launch app", desktop)


if __name__ == "__main__":
    unittest.main()
