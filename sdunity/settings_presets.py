import os
import json

PRESETS_FILE = os.path.join("config", "generator_presets.json")


def load_presets(filepath: str = PRESETS_FILE) -> dict:
    presets = {}
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                presets = json.load(f)
            except json.JSONDecodeError:
                presets = {}
    return presets


def save_presets(presets: dict, filepath: str = PRESETS_FILE) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)


PRESETS = load_presets()


def add_preset(name: str, data: dict) -> None:
    PRESETS[name] = data
    save_presets(PRESETS)


def remove_preset(name: str) -> None:
    if name in PRESETS:
        del PRESETS[name]
        save_presets(PRESETS)
