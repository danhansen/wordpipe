from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import os
import re
import shutil
import subprocess


REMOTE_DESKTOP_IFACE = "org.freedesktop.portal.RemoteDesktop"
GLOBAL_SHORTCUTS_IFACE = "org.freedesktop.portal.GlobalShortcuts"
PORTAL_BUS_NAME = "org.freedesktop.portal.Desktop"
PORTAL_OBJECT_PATH = "/org/freedesktop/portal/desktop"


@dataclass(frozen=True)
class PortalInterface:
    name: str
    available: bool
    methods: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "methods": list(self.methods),
            "error": self.error,
        }


@dataclass(frozen=True)
class ProbeResult:
    session_type: str | None
    gnome_shell: str | None
    commands: dict[str, str | None]
    python_modules: dict[str, bool]
    portals: dict[str, PortalInterface]
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        remote = self.portals.get(REMOTE_DESKTOP_IFACE)
        return bool(
            self.session_type == "wayland"
            and self.python_modules.get("gi", False)
            and remote
            and remote.available
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "session_type": self.session_type,
            "gnome_shell": self.gnome_shell,
            "commands": self.commands,
            "python_modules": self.python_modules,
            "portals": {name: portal.to_dict() for name, portal in self.portals.items()},
            "usable": self.usable,
            "errors": list(self.errors),
        }

    def render_text(self) -> str:
        lines = [
            "Wordpipe capability probe",
            f"  session: {self.session_type or 'unknown'}",
            f"  gnome-shell: {self.gnome_shell or 'not found'}",
            "",
            "Commands:",
        ]
        for name, path in sorted(self.commands.items()):
            lines.append(f"  {name}: {path or 'missing'}")

        lines.append("")
        lines.append("Python modules:")
        for name, available in sorted(self.python_modules.items()):
            state = "ok" if available else "missing"
            lines.append(f"  {name}: {state}")

        lines.append("")
        lines.append("Portals:")
        for portal in self.portals.values():
            state = "available" if portal.available else "unavailable"
            lines.append(f"  {portal.name}: {state}")
            if portal.methods:
                lines.append(f"    methods: {', '.join(portal.methods)}")
            if portal.error:
                lines.append(f"    error: {portal.error}")

        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  {error}")

        return "\n".join(lines)


def _run(command: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _gnome_shell_version() -> str | None:
    if not shutil.which("gnome-shell"):
        return None
    code, stdout, _stderr = _run(["gnome-shell", "--version"])
    if code != 0:
        return None
    return stdout or None


def _introspect_portal(interface: str) -> PortalInterface:
    if shutil.which("busctl"):
        return _introspect_portal_busctl(interface)
    if shutil.which("gdbus"):
        return _introspect_portal_gdbus(interface)
    return PortalInterface(interface, False, error="neither busctl nor gdbus is installed")


def _introspect_portal_busctl(interface: str) -> PortalInterface:
    code, stdout, stderr = _run(
        [
            "busctl",
            "--user",
            "--no-pager",
            "introspect",
            PORTAL_BUS_NAME,
            PORTAL_OBJECT_PATH,
            interface,
        ]
    )
    if code != 0:
        return PortalInterface(interface, False, error=stderr or stdout or "introspection failed")

    methods: list[str] = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "method":
            methods.append(parts[0])

    return PortalInterface(interface, True, tuple(sorted(methods)))


def _introspect_portal_gdbus(interface: str) -> PortalInterface:
    code, stdout, stderr = _run(
        [
            "gdbus",
            "introspect",
            "--session",
            "--dest",
            PORTAL_BUS_NAME,
            "--object-path",
            PORTAL_OBJECT_PATH,
        ]
    )
    if code != 0:
        return PortalInterface(interface, False, error=stderr or stdout or "introspection failed")

    in_interface = False
    in_methods = False
    methods: list[str] = []
    interface_header = f"interface {interface}"
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("interface "):
            in_interface = stripped.startswith(interface_header)
            in_methods = False
            continue
        if not in_interface:
            continue
        if stripped == "methods:":
            in_methods = True
            continue
        if stripped in {"signals:", "properties:"}:
            in_methods = False
            continue
        if in_methods:
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\(", stripped)
            if match:
                methods.append(match.group(1))

    if not methods and interface not in stdout:
        return PortalInterface(interface, False, error="interface not present in gdbus output")
    return PortalInterface(interface, bool(methods), tuple(sorted(methods)))


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_probe() -> ProbeResult:
    commands = {
        "busctl": shutil.which("busctl"),
        "gdbus": shutil.which("gdbus"),
        "gnome-shell": shutil.which("gnome-shell"),
    }
    modules = {
        "gi": _module_available("gi"),
        "numpy": _module_available("numpy"),
        "sherpa_onnx": _module_available("sherpa_onnx"),
        "sounddevice": _module_available("sounddevice"),
    }
    portals = {
        REMOTE_DESKTOP_IFACE: _introspect_portal(REMOTE_DESKTOP_IFACE),
        GLOBAL_SHORTCUTS_IFACE: _introspect_portal(GLOBAL_SHORTCUTS_IFACE),
    }
    session_type = os.environ.get("XDG_SESSION_TYPE")
    errors: list[str] = []
    if session_type != "wayland":
        errors.append("Wordpipe requires a Wayland session for GNOME portal text insertion.")
    if not modules["gi"]:
        errors.append("PyGObject gi is required for GTK and desktop portal D-Bus access.")
    return ProbeResult(
        session_type=session_type,
        gnome_shell=_gnome_shell_version(),
        commands=commands,
        python_modules=modules,
        portals=portals,
        errors=tuple(errors),
    )
