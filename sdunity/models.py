import os
import json
import requests
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from . import config


def load_model_registry(path: str = config.MODEL_REGISTRY_PATH) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


MODEL_REGISTRY = load_model_registry()


def build_model_lookup() -> dict:
    lookup = {}
    for category, models in MODEL_REGISTRY.items():
        for _name, info in models.items():
            filename = os.path.basename(info.get("url", ""))
            if filename:
                lookup[filename] = os.path.join(config.MODELS_DIR, category, filename)
    if os.path.isdir(config.MODELS_DIR):
        for root, _dirs, files in os.walk(config.MODELS_DIR):
            for f in files:
                if f.lower().endswith((".ckpt", ".safetensors", ".bin")):
                    lookup[f] = os.path.join(root, f)
    return lookup


MODEL_LOOKUP = build_model_lookup()

# Cache for loaded pipelines
PIPELINES = {}


def _find_model_info(model_name: str):
    for category, models in MODEL_REGISTRY.items():
        for _name, info in models.items():
            if os.path.basename(info.get("url", "")) == model_name:
                return category, info
    return None, None


def download_model_file(model_name: str, progress=None) -> str:
    category, info = _find_model_info(model_name)
    if not info:
        raise ValueError(f"Unknown model: {model_name}")

    url = info.get("url")
    if not url:
        raise ValueError(f"No download URL for model: {model_name}")

    dest_dir = os.path.join(config.MODELS_DIR, category)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(url))

    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest

    print(f"Downloading {model_name} from {url}")
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        if progress:
            progress(0, desc=f"Downloading {model_name}", total=total)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if progress and total:
                    progress(downloaded, desc=f"Downloading {model_name}", total=total)
        if progress:
            progress(total, desc="Download complete", total=total)
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    return dest


def list_categories() -> list:
    cats = set(MODEL_REGISTRY.keys())
    if os.path.isdir(config.MODELS_DIR):
        for fname in os.listdir(config.MODELS_DIR):
            path = os.path.join(config.MODELS_DIR, fname)
            if os.path.isdir(path):
                cats.add(fname)
            elif fname.lower().endswith((".ckpt", ".safetensors", ".bin")):
                cats.add("Uncategorized")
    return sorted(cats)


def list_models(category=None) -> list:
    names = []
    if category in MODEL_REGISTRY:
        for _name, info in MODEL_REGISTRY[category].items():
            filename = os.path.basename(info.get("url", ""))
            if filename:
                names.append(filename)
    cat_dir = os.path.join(config.MODELS_DIR, category) if category else None
    if cat_dir and os.path.isdir(cat_dir):
        for fname in os.listdir(cat_dir):
            if fname.lower().endswith((".ckpt", ".safetensors", ".bin")):
                if fname not in names:
                    names.append(fname)
    if category == "Uncategorized" and os.path.isdir(config.MODELS_DIR):
        for fname in os.listdir(config.MODELS_DIR):
            path = os.path.join(config.MODELS_DIR, fname)
            if os.path.isfile(path) and fname.lower().endswith((".ckpt", ".safetensors", ".bin")):
                names.append(fname)
    return sorted(names)


def list_loras() -> list:
    if not os.path.isdir(config.LORA_DIR):
        return []
    return [f for f in os.listdir(config.LORA_DIR) if f.lower().endswith(".safetensors")]



def refresh_lists(selected_category=None):
    from gradio import update

    global MODEL_LOOKUP
    MODEL_LOOKUP = build_model_lookup()
    categories = list_categories()
    if selected_category not in categories:
        selected_category = categories[0] if categories else None
    models = list_models(selected_category)
    return (
        update(choices=categories, value=selected_category),
        update(choices=models, value=models[0] if models else None),
        update(choices=list_loras()),
    )



def get_pipeline(model_name: str, progress=None):
    if model_name not in PIPELINES:
        repo = MODEL_LOOKUP.get(model_name)
        if not repo or not os.path.isfile(repo):
            try:
                repo = download_model_file(model_name, progress=progress)
                MODEL_LOOKUP[model_name] = repo
            except Exception as e:
                raise ValueError(f"Unknown model: {model_name}") from e

        if "xl" in model_name.lower() or "xl" in repo.lower():
            pipe = StableDiffusionXLPipeline.from_single_file(
                repo, torch_dtype=torch.float16, variant="fp16"
            )
        else:
            pipe = StableDiffusionPipeline.from_single_file(
                repo, torch_dtype=torch.float16
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        PIPELINES[model_name] = pipe
    return PIPELINES[model_name]
