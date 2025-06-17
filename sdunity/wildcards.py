import os
import re
from . import config

_WILDCARD_CACHE: dict[str, list[str]] = {}

_PATTERN = re.compile(r"__([A-Za-z0-9_-]+)__")


def _load_lines(name: str) -> list[str]:
    path = os.path.join(config.WILDCARDS_DIR, f"{name}.txt")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _lines(name: str) -> list[str]:
    if name not in _WILDCARD_CACHE:
        _WILDCARD_CACHE[name] = _load_lines(name)
    return _WILDCARD_CACHE[name]


def apply(text: str, index: int) -> str:
    """Replace wildcard tokens in ``text`` using ``index`` as line selector."""

    def repl(match: re.Match) -> str:
        name = match.group(1)
        lines = _lines(name)
        if not lines:
            return match.group(0)
        if index < len(lines):
            return lines[index]
        return lines[-1]

    return _PATTERN.sub(repl, text)
