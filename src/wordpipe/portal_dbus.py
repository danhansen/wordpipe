from __future__ import annotations

from typing import Callable

from .probe import PORTAL_BUS_NAME, PORTAL_OBJECT_PATH


REQUEST_IFACE = "org.freedesktop.portal.Request"
SESSION_IFACE = "org.freedesktop.portal.Session"
PortalSignalCallback = Callable[[tuple[object, ...], str], None]


class GioPortalProxy:
    def __init__(self, interface: str, object_path: str = PORTAL_OBJECT_PATH) -> None:
        try:
            import gi

            gi.require_version("Gio", "2.0")
            from gi.repository import Gio, GLib
        except (ImportError, ValueError) as exc:
            raise RuntimeError("PyGObject Gio is required for desktop portal D-Bus access") from exc

        self.Gio = Gio
        self.GLib = GLib
        self._bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        self._proxy = Gio.DBusProxy.new_sync(
            self._bus,
            Gio.DBusProxyFlags.NONE,
            None,
            PORTAL_BUS_NAME,
            object_path,
            interface,
            None,
        )

    def call(self, method: str, signature: str, values: tuple[object, ...]) -> tuple[object, ...]:
        result = self._proxy.call_sync(
            method,
            self.GLib.Variant(signature, values),
            self.Gio.DBusCallFlags.NONE,
            -1,
            None,
        )
        return result.unpack()

    def request(
        self,
        method: str,
        signature: str,
        values: tuple[object, ...],
        *,
        app_id_error: str | None = None,
        timeout_seconds: int = 30,
    ) -> dict[str, object]:
        loop = self.GLib.MainLoop()
        response: dict[str, object] = {}
        error: list[BaseException] = []
        expected_handle: list[str] = []
        timeout_active = [False]
        timeout_id: int | None = None

        def on_response(
            _connection,
            _sender,
            object_path,
            _interface,
            _signal,
            parameters,
        ) -> None:  # type: ignore[no-untyped-def]
            if expected_handle and str(object_path) != expected_handle[0]:
                return
            code, results = parameters.unpack()
            if int(code) != 0:
                error.append(RuntimeError(f"portal request failed with response code {int(code)}"))
            else:
                response.update(_unpack_variant_dict(results))
            loop.quit()

        def on_timeout() -> bool:
            timeout_active[0] = False
            error.append(TimeoutError(f"portal request {method} timed out after {timeout_seconds}s"))
            loop.quit()
            return False

        subscription = self._bus.signal_subscribe(
            PORTAL_BUS_NAME,
            REQUEST_IFACE,
            "Response",
            None,
            None,
            self.Gio.DBusSignalFlags.NONE,
            on_response,
        )
        try:
            try:
                (handle,) = self.call(method, signature, values)
            except self.GLib.GError as exc:
                message = str(exc)
                if app_id_error is not None and "An app id is required" in message:
                    raise RuntimeError(app_id_error) from exc
                raise
            expected_handle.append(str(handle))
            if timeout_seconds > 0:
                timeout_active[0] = True
                timeout_id = self.GLib.timeout_add_seconds(timeout_seconds, on_timeout)
            loop.run()
            if error:
                raise error[0]
            return response
        finally:
            if timeout_active[0]:
                assert timeout_id is not None
                self.GLib.source_remove(timeout_id)
            self._bus.signal_unsubscribe(subscription)

    def subscribe_signal(
        self,
        interface: str,
        signal_name: str,
        callback: PortalSignalCallback,
    ) -> int:
        def on_signal(
            _connection,
            _sender,
            object_path,
            _interface,
            _signal,
            parameters,
        ) -> None:  # type: ignore[no-untyped-def]
            callback(parameters.unpack(), str(object_path))

        return int(
            self._bus.signal_subscribe(
                PORTAL_BUS_NAME,
                interface,
                signal_name,
                None,
                None,
                self.Gio.DBusSignalFlags.NONE,
                on_signal,
            )
        )

    def unsubscribe_signal(self, subscription: int) -> None:
        self._bus.signal_unsubscribe(subscription)

    def close_session(self, session_handle: str) -> None:
        proxy = self.Gio.DBusProxy.new_sync(
            self._bus,
            self.Gio.DBusProxyFlags.NONE,
            None,
            PORTAL_BUS_NAME,
            session_handle,
            SESSION_IFACE,
            None,
        )
        proxy.call_sync("Close", None, self.Gio.DBusCallFlags.NONE, -1, None)


def variant_options(options: dict[str, object]) -> dict[str, object]:
    from gi.repository import GLib

    variants: dict[str, object] = {}
    for key, value in options.items():
        if hasattr(value, "unpack"):
            variants[key] = value
        elif isinstance(value, bool):
            variants[key] = GLib.Variant("b", value)
        elif isinstance(value, int):
            variants[key] = GLib.Variant("i", value)
        else:
            variants[key] = GLib.Variant("s", str(value))
    return variants


def variant_uint32(value: int):
    from gi.repository import GLib

    return GLib.Variant("u", value)


def _unpack_variant_dict(values: dict[str, object]) -> dict[str, object]:
    unpacked: dict[str, object] = {}
    for key, value in values.items():
        unpacked[str(key)] = value.unpack() if hasattr(value, "unpack") else value
    return unpacked
