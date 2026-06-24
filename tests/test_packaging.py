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

    def test_flatpak_dev_runner_mounts_source_checkout(self) -> None:
        runner = (ROOT / "scripts/wordpipe-flatpak-dev").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "APP_ID=${WORDPIPE_FLATPAK_APP_ID:-dev.wordpipe.Wordpipe}",
            runner,
        )
        self.assertIn('--filesystem="$ROOT:ro"', runner)
        self.assertIn('--env=PYTHONPATH="$ROOT/src"', runner)
        self.assertIn("--command=python3", runner)
        self.assertIn("-m wordpipe", runner)

    def test_python_flatpak_module_uses_launcher_file_source(self) -> None:
        manifest = (ROOT / "packaging/flatpak/dev.wordpipe.Wordpipe.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "path: ../../packaging/flatpak/wordpipe-flatpak-launch",
            manifest,
        )
        self.assertNotIn("path: ../../packaging/flatpak\n", manifest)


if __name__ == "__main__":
    unittest.main()
