from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from typing import Sequence

from wordpipe.shortcuts import (
    DEFAULT_SHORTCUT_BINDING,
    FLATPAK_SHORTCUT_PATH,
    LOCAL_SHORTCUT_PATH,
    flatpak_shortcut_spec,
    install_shortcut,
    local_shortcut_spec,
    read_shortcut_status,
    remove_shortcut_paths,
)


class FakeGSettings:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {
            (
                "org.gnome.settings-daemon.plugins.media-keys",
                "custom-keybindings",
            ): "@as []",
        }
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(command)
        self.commands.append(args)
        if args[0] != "gsettings":
            return subprocess.CompletedProcess(args, 127, "", "not found")
        if args[1] == "get":
            value = self.values.get((args[2], args[3]), "''")
            return subprocess.CompletedProcess(args, 0, value + "\n", "")
        if args[1] == "set":
            self.values[(args[2], args[3])] = args[4]
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 2, "", "unsupported")


class ShortcutBackendTests(unittest.TestCase):
    def test_flatpak_spec_uses_toggle_command(self) -> None:
        spec = flatpak_shortcut_spec(app_id="dev.example.App")

        self.assertEqual(spec.path, FLATPAK_SHORTCUT_PATH)
        self.assertEqual(
            spec.command,
            "flatpak run dev.example.App voice-keyboard-toggle --start-if-needed",
        )
        self.assertEqual(spec.binding, DEFAULT_SHORTCUT_BINDING)

    def test_local_spec_uses_wordpipe_dev_helper(self) -> None:
        spec = local_shortcut_spec(Path("/repo"))

        self.assertEqual(spec.path, LOCAL_SHORTCUT_PATH)
        self.assertEqual(
            spec.command,
            "/repo/scripts/wordpipe-dev voice-keyboard-toggle --start-if-needed",
        )

    def test_status_reports_missing_shortcut(self) -> None:
        fake = FakeGSettings()
        spec = flatpak_shortcut_spec()

        status = read_shortcut_status(spec, runner=fake)

        self.assertFalse(status.present)
        self.assertFalse(status.matches)
        self.assertEqual(status.summary, "not installed")

    def test_install_adds_path_and_sets_shortcut_fields(self) -> None:
        fake = FakeGSettings()
        spec = flatpak_shortcut_spec(binding="<Super>d")

        status = install_shortcut(spec, runner=fake)

        self.assertTrue(status.present)
        self.assertTrue(status.matches)
        self.assertEqual(status.binding, "<Super>d")
        paths = fake.values[
            ("org.gnome.settings-daemon.plugins.media-keys", "custom-keybindings")
        ]
        self.assertEqual(paths, f"['{FLATPAK_SHORTCUT_PATH}']")

    def test_install_repairs_existing_shortcut_without_duplicating_path(self) -> None:
        fake = FakeGSettings()
        spec = flatpak_shortcut_spec(binding="<Super>d")
        fake.values[
            ("org.gnome.settings-daemon.plugins.media-keys", "custom-keybindings")
        ] = f"['{FLATPAK_SHORTCUT_PATH}']"
        fake.values[(spec.schema_with_path, "name")] = "'Old'"
        fake.values[(spec.schema_with_path, "command")] = "'old-command'"
        fake.values[(spec.schema_with_path, "binding")] = "'<Super>o'"

        status = install_shortcut(spec, runner=fake)

        self.assertTrue(status.matches)
        self.assertEqual(status.name, "Wordpipe Dictation")
        self.assertEqual(status.command, spec.command)
        self.assertEqual(status.binding, "<Super>d")
        set_path_commands = [
            command
            for command in fake.commands
            if command[1:4]
            == ["set", "org.gnome.settings-daemon.plugins.media-keys", "custom-keybindings"]
        ]
        self.assertEqual(set_path_commands, [])

    def test_remove_shortcut_paths_removes_only_requested_paths(self) -> None:
        fake = FakeGSettings()
        other_path = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/other/"
        fake.values[
            ("org.gnome.settings-daemon.plugins.media-keys", "custom-keybindings")
        ] = f"['{LOCAL_SHORTCUT_PATH}', '{other_path}', '{FLATPAK_SHORTCUT_PATH}']"
        fake.values[
            (
                f"org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:"
                f"{LOCAL_SHORTCUT_PATH}",
                "binding",
            )
        ] = "'<Control><Alt>space'"
        fake.values[
            (
                f"org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:"
                f"{FLATPAK_SHORTCUT_PATH}",
                "binding",
            )
        ] = "'<Control><Alt>g'"

        removed = remove_shortcut_paths(
            (LOCAL_SHORTCUT_PATH, FLATPAK_SHORTCUT_PATH),
            runner=fake,
        )

        self.assertEqual(removed, (LOCAL_SHORTCUT_PATH, FLATPAK_SHORTCUT_PATH))
        self.assertEqual(
            fake.values[
                ("org.gnome.settings-daemon.plugins.media-keys", "custom-keybindings")
            ],
            f"['{other_path}']",
        )
        self.assertEqual(
            fake.values[
                (
                    f"org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:"
                    f"{LOCAL_SHORTCUT_PATH}",
                    "binding",
                )
            ],
            "''",
        )
        self.assertEqual(
            fake.values[
                (
                    f"org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:"
                    f"{FLATPAK_SHORTCUT_PATH}",
                    "binding",
                )
            ],
            "''",
        )

    def test_failed_gsettings_command_raises_runtime_error(self) -> None:
        def fail(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(list(command), 1, "", "no dconf")

        with self.assertRaisesRegex(RuntimeError, "no dconf"):
            read_shortcut_status(flatpak_shortcut_spec(), runner=fail)


if __name__ == "__main__":
    unittest.main()
