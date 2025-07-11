import os
import json
import threading
from typing import Dict, Tuple

from . import config, civitai

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

        downloads = {}
        for fname, url in (data.get("checkpoint_downloads") or {}).items():
            downloads[fname] = (config.MODELS_DIR, url)
        for fname, url in (data.get("lora_downloads") or {}).items():
            downloads[fname] = (config.LORA_DIR, url)
        for fname, url in (data.get("embeddings_downloads") or {}).items():
            dest = os.path.join(config.BASE_DIR, "embeddings")
            downloads[fname] = (dest, url)
        if downloads:
            cfg["downloads"] = downloads
        presets[os.path.splitext(fname)[0]] = cfg
    return presets


# Load presets at import time
PRESETS = load_presets()


def _file_exists(directory: str, name: str) -> bool:
    for root, _dirs, files in os.walk(directory):
        if name in files:
            return True
    return False


def ensure_preset_assets(name: str) -> str:
    """Download missing files referenced by a preset in the background."""
    data = PRESETS.get(name)
    if not data:
        return ""
    downloads = data.get("downloads") or {}
    messages = []

    for fname, (dest, url) in downloads.items():
        if not _file_exists(dest, fname):

            def _dl(u=url, d=dest, f=fname):
                try:
                    civitai.download_model(u, d)
                except Exception as e:  # pragma: no cover - network
                    print(f"Download failed for {f}:", e)

            threading.Thread(target=_dl, daemon=True).start()
            messages.append(f"Downloading {fname} in background...")

    return "\n".join(messages)
