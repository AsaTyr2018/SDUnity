import os
import bisect
import re

_TAG_FILE = os.path.join("data", "all_tags.csv")

# Load and sort tags on import for fast lookup
if os.path.isfile(_TAG_FILE):
    with open(_TAG_FILE, "r", encoding="utf-8") as f:
        _TAGS = sorted({line.strip().rstrip(',') for line in f if line.strip()})
else:
    _TAGS = []


def suggestions(prefix: str, limit: int = 10) -> list[str]:
    """Return a list of tag suggestions for the given prefix."""
    if not prefix:
        return []
    prefix = prefix.lower()
    idx = bisect.bisect_left(_TAGS, prefix)
    result = []
    while idx < len(_TAGS):
        tag = _TAGS[idx]
        if not tag.startswith(prefix):
            break
        result.append(tag)
        if len(result) >= limit:
            break
        idx += 1
    return result


def _last_word(text: str) -> str:
    """Extract the last word from a comma/space separated string."""
    parts = re.split(r"[\s,]+", text.strip())
    return parts[-1] if parts else ""


def suggestions_from_prompt(prompt: str, limit: int = 10) -> list[str]:
    """Return suggestions based on the last word of the prompt."""
    return suggestions(_last_word(prompt), limit=limit)


def apply_suggestion(prompt: str, suggestion: str) -> str:
    """Replace the last word in the prompt with the suggestion."""
    if not suggestion:
        return prompt
    tokens = re.split(r"([,\s]+)", prompt)
    # Find last non-delimiter token
    for i in range(len(tokens) - 1, -1, -1):
        if not re.fullmatch(r"[,\s]+", tokens[i]):
            tokens[i] = suggestion
            break
    else:
        tokens.append(suggestion)
    return "".join(tokens)
