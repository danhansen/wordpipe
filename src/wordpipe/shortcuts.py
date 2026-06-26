from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


MEDIA_KEYS_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys"
CUSTOM_KEYBINDING_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding"
CUSTOM_KEYBINDINGS_KEY = "custom-keybindings"
DEFAULT_SHORTCUT_NAME = "Wordpipe Dictation"
DEFAULT_SHORTCUT_BINDING = "<Control><Alt>space"
DEFAULT_FLATPAK_APP_ID = "dev.wordpipe.Wordpipe"
FLATPAK_SHORTCUT_PATH = (
    "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/wordpipe-flatpak/"
)
LOCAL_SHORTCUT_PATH = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/wordpipe/"


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ShortcutSpec:
    path: str
    name: str
    command: str
    binding: str

    @property
    def schema_with_path(self) -> str:
        return f"{CUSTOM_KEYBINDING_SCHEMA}:{self.path}"


@dataclass(frozen=True)
class ShortcutStatus:
    spec: ShortcutSpec
    configured_paths: tuple[str, ...]
    present: bool
    name: str = ""
    command: str = ""
    binding: str = ""

    @property
    def matches(self) -> bool:
        return (
            self.present
            and self.name == self.spec.name
            and self.command == self.spec.command
            and self.binding == self.spec.binding
        )

    @property
    def summary(self) -> str:
        if self.matches:
            return f"installed: {self.binding} -> {self.command}"
        if self.present:
            return (
                "configured differently: "
                f"{self.binding or '<unset>'} -> {self.command or '<unset>'}"
            )
        return "not installed"


def flatpak_shortcut_spec(
    *,
    app_id: str = DEFAULT_FLATPAK_APP_ID,
    binding: str = DEFAULT_SHORTCUT_BINDING,
) -> ShortcutSpec:
    return ShortcutSpec(
        path=FLATPAK_SHORTCUT_PATH,
        name=DEFAULT_SHORTCUT_NAME,
        command=f"flatpak run {app_id} voice-keyboard-toggle --start-if-needed",
        binding=binding,
    )


def local_shortcut_spec(
    root: Path,
    *,
    binding: str = DEFAULT_SHORTCUT_BINDING,
) -> ShortcutSpec:
    return ShortcutSpec(
        path=LOCAL_SHORTCUT_PATH,
        name=DEFAULT_SHORTCUT_NAME,
        command=f"{root / 'scripts' / 'wordpipe-dev'} voice-keyboard-toggle --start-if-needed",
        binding=binding,
    )


def read_shortcut_status(
    spec: ShortcutSpec,
    *,
    runner: CommandRunner | None = None,
) -> ShortcutStatus:
    paths = tuple(_read_custom_keybinding_paths(runner=runner))
    if spec.path not in paths:
        return ShortcutStatus(spec=spec, configured_paths=paths, present=False)
    return ShortcutStatus(
        spec=spec,
        configured_paths=paths,
        present=True,
        name=_gsettings_get_string(spec.schema_with_path, "name", runner=runner),
        command=_gsettings_get_string(spec.schema_with_path, "command", runner=runner),
        binding=_gsettings_get_string(spec.schema_with_path, "binding", runner=runner),
    )


def install_shortcut(
    spec: ShortcutSpec,
    *,
    runner: CommandRunner | None = None,
) -> ShortcutStatus:
    paths = list(_read_custom_keybinding_paths(runner=runner))
    if spec.path not in paths:
        paths.append(spec.path)
        _gsettings_set(MEDIA_KEYS_SCHEMA, CUSTOM_KEYBINDINGS_KEY, _format_gvariant_list(paths), runner)
    _gsettings_set(spec.schema_with_path, "name", _quote_gvariant_string(spec.name), runner)
    _gsettings_set(spec.schema_with_path, "command", _quote_gvariant_string(spec.command), runner)
    _gsettings_set(spec.schema_with_path, "binding", _quote_gvariant_string(spec.binding), runner)
    return read_shortcut_status(spec, runner=runner)


def remove_shortcut_paths(
    paths_to_remove: Sequence[str],
    *,
    runner: CommandRunner | None = None,
) -> tuple[str, ...]:
    paths = list(_read_custom_keybinding_paths(runner=runner))
    remove_set = set(paths_to_remove)
    remaining = [path for path in paths if path not in remove_set]
    removed = tuple(path for path in paths if path in remove_set)

    if remaining != paths:
        _gsettings_set(
            MEDIA_KEYS_SCHEMA,
            CUSTOM_KEYBINDINGS_KEY,
            _format_gvariant_list(remaining),
            runner,
        )

    for path in paths_to_remove:
        schema = f"{CUSTOM_KEYBINDING_SCHEMA}:{path}"
        _gsettings_set(schema, "binding", _quote_gvariant_string(""), runner)

    return removed


def _read_custom_keybinding_paths(*, runner: CommandRunner | None = None) -> list[str]:
    raw = _gsettings_get(MEDIA_KEYS_SCHEMA, CUSTOM_KEYBINDINGS_KEY, runner=runner)
    return _parse_gvariant_string_list(raw)


def _gsettings_get(
    schema: str,
    key: str,
    *,
    runner: CommandRunner | None = None,
) -> str:
    result = _run_gsettings(["get", schema, key], runner)
    return result.stdout.strip()


def _gsettings_get_string(
    schema: str,
    key: str,
    *,
    runner: CommandRunner | None = None,
) -> str:
    raw = _gsettings_get(schema, key, runner=runner)
    value = ast.literal_eval(raw)
    if not isinstance(value, str):
        raise RuntimeError(f"gsettings {schema} {key} is not a string: {raw}")
    return value


def _gsettings_set(
    schema: str,
    key: str,
    value: str,
    runner: CommandRunner | None,
) -> None:
    _run_gsettings(["set", schema, key, value], runner)


def _run_gsettings(
    args: Sequence[str],
    runner: CommandRunner | None,
) -> subprocess.CompletedProcess[str]:
    command = ["gsettings", *args]
    if runner is None:
        runner = _default_runner
    result = runner(command)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return result


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _parse_gvariant_string_list(raw: str) -> list[str]:
    text = raw.strip()
    if text in {"@as []", "[]"}:
        return []
    value = ast.literal_eval(text)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"gsettings custom-keybindings is not a string array: {raw}")
    return value


def _format_gvariant_list(values: Sequence[str]) -> str:
    return "[" + ", ".join(_quote_gvariant_string(value) for value in values) + "]"


def _quote_gvariant_string(value: str) -> str:
    return repr(value)
