import os
import json
from typing import Dict

STYLE_DIR = "styles"


def load_presets(directory: str = STYLE_DIR) -> Dict[str, dict]:
    """Load style presets from JSON files in ``directory``.

    Each JSON file represents a category and contains a list of objects with
    ``name``, ``prompt`` and ``negative_prompt`` fields. The returned mapping
    uses the format ``"<category> | <name>"`` as keys and stores a dictionary
    with ``prompt`` and ``negative_prompt`` values.
    """
    presets: Dict[str, dict] = {}
    if not os.path.isdir(directory):
        return presets

    for fname in os.listdir(directory):
        if not fname.lower().endswith(".json"):
            continue
        category = os.path.splitext(fname)[0]
        path = os.path.join(directory, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            prompt = item.get("prompt", "")
            negative = item.get("negative_prompt", "")
            display = f"{category} | {name}"
            presets[display] = {"prompt": prompt, "negative_prompt": negative}
    return presets


PRESETS = load_presets()
