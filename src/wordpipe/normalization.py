from __future__ import annotations


PUNCTUATION_COMMANDS = {
    ("comma",): ",",
    ("period",): ".",
    ("full", "stop"): ".",
    ("question", "mark"): "?",
    ("exclamation", "point"): "!",
    ("exclamation", "mark"): "!",
    ("colon",): ":",
    ("semicolon",): ";",
}

BREAK_COMMANDS = {
    ("new", "line"): "\n",
    ("newline",): "\n",
    ("new", "paragraph"): "\n\n",
}

MAX_COMMAND_WORDS = max(
    len(command) for command in tuple(PUNCTUATION_COMMANDS) + tuple(BREAK_COMMANDS)
)


def normalize_spoken_punctuation(text: str) -> str:
    words = text.split()
    output = ""
    index = 0
    while index < len(words):
        command, consumed = _match_command(words, index)
        if command is None:
            output = _append_word(output, words[index])
            index += 1
            continue

        if command in {",", ".", "?", "!", ":", ";"}:
            output = output.rstrip() + command + " "
        else:
            output = output.rstrip() + command
        index += consumed

    return output.strip(" ")


def _match_command(words: list[str], index: int) -> tuple[str | None, int]:
    remaining = len(words) - index
    for size in range(min(MAX_COMMAND_WORDS, remaining), 0, -1):
        candidate = tuple(word.lower() for word in words[index : index + size])
        if candidate in PUNCTUATION_COMMANDS:
            return PUNCTUATION_COMMANDS[candidate], size
        if candidate in BREAK_COMMANDS:
            return BREAK_COMMANDS[candidate], size
    return None, 0


def _append_word(output: str, word: str) -> str:
    if not output or output.endswith(("\n", " ")):
        return output + word + " "
    return output + " " + word + " "
