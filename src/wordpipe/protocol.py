from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal


CommandName = Literal["start", "stop", "shutdown"]
EventName = Literal["ready", "listening", "stopped", "partial", "commit", "error"]


@dataclass(frozen=True)
class Command:
    command: CommandName


@dataclass(frozen=True)
class Event:
    event: EventName
    text: str | None = None
    message: str | None = None
    data: dict[str, Any] | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {"event": self.event}
        if self.text is not None:
            payload["text"] = self.text
        if self.message is not None:
            payload["message"] = self.message
        if self.data is not None:
            payload["data"] = self.data
        return json.dumps(payload, sort_keys=True)


def parse_command(line: str) -> Command:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON command: {exc}") from exc

    name = payload.get("command")
    if name not in {"start", "stop", "shutdown"}:
        raise ValueError(f"unknown command: {name!r}")
    return Command(command=name)


def event(name: EventName, **kwargs: Any) -> Event:
    return Event(event=name, **kwargs)
