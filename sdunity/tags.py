import os
import bisect
import re
import csv
from rapidfuzz import process

_TAG_FILE = os.path.join("data", "all_tags.csv")


def _load_csv_tags(path: str) -> list[str]:
    """Return tags from a simple CSV file with one tag per line."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().rstrip(",") for line in f if line.strip()]


def _save_csv_tags(tags: set[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in sorted(tags):
            f.write(f"{t},\n")


def _load_tagcomplete_csv(path: str) -> set[str]:
    """Parse a tagcomplete CSV file and return tag names including synonyms."""
    result: set[str] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            tag = row[0].strip()
            if tag:
                result.add(tag)
            if len(row) >= 4 and row[3].strip():
                syns = [s.strip() for s in row[3].split(',') if s.strip()]
                result.update(syns)
    return result


def load_tagcomplete_tags(repo_path: str) -> set[str]:
    """Load all tags from a clone of a1111-sd-webui-tagcomplete."""
    tags_dir = os.path.join(repo_path, "tags")
    tags: set[str] = set()
    for root, _dirs, files in os.walk(tags_dir):
        for fname in files:
            if fname.lower().endswith(".csv"):
                tags.update(_load_tagcomplete_csv(os.path.join(root, fname)))
    return tags


def update_dataset_from_tagcomplete(repo_path: str, output_path: str = _TAG_FILE) -> None:
    """Generate the dataset from the tagcomplete repository."""
    tags = load_tagcomplete_tags(repo_path)
    _save_csv_tags(tags, output_path)


# Load and sort tags on import for fast lookup
_TAGS = sorted(set(_load_csv_tags(_TAG_FILE)))


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
    if len(result) < limit:
        # Fuzzy search for additional results
        fuzzy = process.extract(prefix, _TAGS, limit=limit - len(result))
        result.extend([m[0] for m in fuzzy if m[0] not in result])
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
