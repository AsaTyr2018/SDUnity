import os

PRESETS_FILE = "presets.txt"


def load_presets(filepath: str = PRESETS_FILE) -> dict:
    """Load prompt enhancement presets from a pipe-separated file."""
    presets = {}
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines[1:]:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                display = f"{parts[0]} | {parts[1]}"
                presets[display] = parts[2]
    return presets


PRESETS = load_presets()
