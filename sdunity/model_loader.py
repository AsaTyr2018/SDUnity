import os
import json
from typing import Dict, Tuple

# Directory containing model loader presets
PRESET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets")


def _parse_aspect_ratio(value: str) -> Tuple[int | None, int | None]:
    try:
        w, h = value.split("*")
        return int(w), int(h)
    except Exception:
        return None, None


def load_presets(directory: str = PRESET_DIR) -> Dict[str, dict]:
    """Load model loader presets from ``directory``.

    Parameters
    ----------
    directory : str
        Path to the folder containing preset JSON files.

    Returns
    -------
    Dict[str, dict]
        Mapping of preset names to their configuration dictionaries.
    """
    presets: Dict[str, dict] = {}
    if not os.path.isdir(directory):
        return presets

    for fname in os.listdir(directory):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(directory, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        cfg = {
            "model": data.get("default_model"),
            "loras": [
                item[1]
                for item in data.get("default_loras", [])
                if isinstance(item, list) and item and item[0] and item[1] != "None"
            ],
            "lora_weights": [
                float(item[2])
                for item in data.get("default_loras", [])
                if isinstance(item, list) and item and item[0] and item[1] != "None"
            ],
            "guidance_scale": data.get("default_cfg_scale"),
            "prompt": data.get("default_prompt"),
            "negative_prompt": data.get("default_prompt_negative"),
        }
        w, h = _parse_aspect_ratio(str(data.get("default_aspect_ratio", "")))
        if w and h:
            cfg["width"] = w
            cfg["height"] = h
        step = data.get("default_overwrite_step")
        if isinstance(step, int) and step > 0:
            cfg["steps"] = step
        presets[os.path.splitext(fname)[0]] = cfg
    return presets


# Load presets at import time
PRESETS = load_presets()
