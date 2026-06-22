from __future__ import annotations

import json
import unittest

from wordpipe.protocol import event, parse_command


class ProtocolTests(unittest.TestCase):
    def test_parse_command_accepts_known_command(self) -> None:
        command = parse_command('{"command": "start"}')

        self.assertEqual(command.command, "start")

    def test_parse_command_rejects_unknown_command(self) -> None:
        with self.assertRaises(ValueError):
            parse_command('{"command": "pause"}')

    def test_event_serializes_without_null_fields(self) -> None:
        payload = json.loads(event("commit", text="hello").to_json())

        self.assertEqual(payload, {"event": "commit", "text": "hello"})


if __name__ == "__main__":
    unittest.main()
