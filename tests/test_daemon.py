from __future__ import annotations

import unittest

from wordpipe.daemon import format_committed_text


class DaemonTests(unittest.TestCase):
    def test_commit_formatter_strips_and_adds_space(self) -> None:
        self.assertEqual(format_committed_text(" hello "), "hello ")

    def test_commit_formatter_ignores_empty_text(self) -> None:
        self.assertEqual(format_committed_text("   "), "")


if __name__ == "__main__":
    unittest.main()
