from __future__ import annotations

import unittest

from wordpipe.normalization import normalize_spoken_punctuation


class NormalizationTests(unittest.TestCase):
    def test_spoken_punctuation_attaches_to_previous_word(self) -> None:
        self.assertEqual(
            normalize_spoken_punctuation("hello comma world period"),
            "hello, world.",
        )

    def test_multi_word_punctuation_commands(self) -> None:
        self.assertEqual(
            normalize_spoken_punctuation("is this working question mark yes exclamation point"),
            "is this working? yes!",
        )

    def test_line_break_commands(self) -> None:
        self.assertEqual(
            normalize_spoken_punctuation("hello new line world new paragraph done"),
            "hello\nworld\n\ndone",
        )


if __name__ == "__main__":
    unittest.main()
