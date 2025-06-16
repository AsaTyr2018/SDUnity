import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from typing import Optional

from . import config


def build_model_lookup() -> dict:
    lookup = {}
    if os.path.isdir(config.MODELS_DIR):
        for root, _dirs, files in os.walk(config.MODELS_DIR):
            for f in files:
                if f.lower().endswith((".ckpt", ".safetensors", ".bin")):
                    lookup[f] = os.path.join(root, f)
    return lookup


MODEL_LOOKUP = build_model_lookup()


def build_lora_lookup() -> dict:
    lookup = {}
    if os.path.isdir(config.LORA_DIR):
        for root, _dirs, files in os.walk(config.LORA_DIR):
            for f in files:
                if f.lower().endswith(".safetensors"):
                    lookup[f] = os.path.join(root, f)
    return lookup


LORA_LOOKUP = build_lora_lookup()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_category(model_name: str) -> Optional[str]:
    """Return the category directory for a model name."""
    path = MODEL_LOOKUP.get(model_name)
    if not path:
        return None
    parent = os.path.dirname(path)
    root = os.path.abspath(config.MODELS_DIR)
    if os.path.abspath(parent) == root:
        return "Uncategorized"
    return os.path.relpath(parent, root)


def _lora_category(lora_name: str) -> Optional[str]:
    path = LORA_LOOKUP.get(lora_name)
    if not path:
        return None
    parent = os.path.dirname(path)
    root = os.path.abspath(config.LORA_DIR)
    if os.path.abspath(parent) == root:
        return "Uncategorized"
    return os.path.relpath(parent, root)

# Cache for backend instances indexed by category
BACKENDS = {}


class BaseBackend:
    """Basic backend wrapper handling pipeline loading."""

    pipeline_cls = StableDiffusionPipeline

    def __init__(self):
        self.pipelines = {}

    def build_pipeline(self, repo: str):
        return self.pipeline_cls.from_single_file(repo, torch_dtype=torch.float16)

    def get_pipeline(self, model_name: str, progress=None):
        if model_name not in self.pipelines:
            repo = MODEL_LOOKUP.get(model_name)
            if not repo or not os.path.isfile(repo):
                raise ValueError(f"Unknown model: {model_name}")

            pipe = self.build_pipeline(repo)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe.to(device)
            self.pipelines[model_name] = pipe
        return self.pipelines[model_name]


class SD15Backend(BaseBackend):
    pipeline_cls = StableDiffusionPipeline


class SDXLBackend(BaseBackend):
    pipeline_cls = StableDiffusionXLPipeline

    def build_pipeline(self, repo: str):
        return self.pipeline_cls.from_single_file(
            repo, torch_dtype=torch.float16, variant="fp16"
        )


def get_backend(category: str) -> BaseBackend:
    """Return a backend suitable for the given model category."""
    key = category or "SD15"
    if "xl" in key.lower():
        backend_cls = SDXLBackend
    else:
        backend_cls = SD15Backend
    if key not in BACKENDS:
        BACKENDS[key] = backend_cls()
    return BACKENDS[key]




def list_categories() -> list:
    cats = set()
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


def list_lora_categories() -> list:
    cats = set()
    if os.path.isdir(config.LORA_DIR):
        for fname in os.listdir(config.LORA_DIR):
            path = os.path.join(config.LORA_DIR, fname)
            if os.path.isdir(path):
                cats.add(fname)
            elif fname.lower().endswith(".safetensors"):
                cats.add("Uncategorized")
    return sorted(cats)


def list_loras(category: str | None = None) -> list:
    names = []
    cat_dir = os.path.join(config.LORA_DIR, category) if category else None
    if cat_dir and os.path.isdir(cat_dir):
        for fname in os.listdir(cat_dir):
            if fname.lower().endswith(".safetensors"):
                if fname not in names:
                    names.append(fname)
    if category == "Uncategorized" and os.path.isdir(config.LORA_DIR):
        for fname in os.listdir(config.LORA_DIR):
            path = os.path.join(config.LORA_DIR, fname)
            if os.path.isfile(path) and fname.lower().endswith(".safetensors"):
                names.append(fname)
    if category is None:
        # return all loras
        for root, _dirs, files in os.walk(config.LORA_DIR):
            for f in files:
                if f.lower().endswith(".safetensors") and f not in names:
                    names.append(f)
    return sorted(names)



def refresh_lists(selected_category=None):
    from gradio import update

    global MODEL_LOOKUP, LORA_LOOKUP
    MODEL_LOOKUP = build_model_lookup()
    LORA_LOOKUP = build_lora_lookup()
    categories = list_categories()
    if selected_category not in categories:
        selected_category = categories[0] if categories else None
    models = list_models(selected_category)
    return (
        update(choices=categories, value=selected_category),
        update(choices=models, value=models[0] if models else None),
        update(choices=list_loras()),
    )



def get_pipeline(model_name: str, progress=None, category: Optional[str] = None):
    """Return a cached pipeline for the given model using its category."""
    if category is None:
        category = _model_category(model_name)
    backend = get_backend(category)
    return backend.get_pipeline(model_name, progress=progress)


def remove_model_file(model_name: str, category: Optional[str] = None) -> bool:
    """Delete a model file from disk and update caches.

    Parameters
    ----------
    model_name:
        Filename of the model to remove.
    category:
        Optional category the model currently resides in. If omitted the
        function will attempt to locate it automatically.

    Returns
    -------
    bool
        ``True`` if the file was removed, ``False`` if it was not found.
    """

    if category is None:
        category = _model_category(model_name)

    path = MODEL_LOOKUP.get(model_name)
    if category and not path:
        path = os.path.join(config.MODELS_DIR, category, model_name)

    if not path or not os.path.isfile(path):
        return False

    try:
        os.remove(path)
    finally:
        if model_name in MODEL_LOOKUP:
            del MODEL_LOOKUP[model_name]
        for backend in BACKENDS.values():
            backend.pipelines.pop(model_name, None)
    return True


def move_model_file(
    model_name: str,
    new_category: str,
    current_category: Optional[str] = None,
) -> str:
    """Move a model file to a different category directory.

    Parameters
    ----------
    model_name:
        Filename of the model to move.
    new_category:
        Destination category name.
    current_category:
        Optional current category of the model. If not provided the file will
        be located automatically.

    Returns
    -------
    str
        Path to the relocated model file.
    """

    if current_category is None:
        current_category = _model_category(model_name)

    src = MODEL_LOOKUP.get(model_name)
    if current_category and not src:
        src = os.path.join(config.MODELS_DIR, current_category, model_name)

    if not src or not os.path.isfile(src):
        raise FileNotFoundError(model_name)

    dest_dir = os.path.join(config.MODELS_DIR, new_category)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, model_name)

    if os.path.abspath(src) == os.path.abspath(dest):
        return dest

    os.rename(src, dest)

    MODEL_LOOKUP[model_name] = dest
    for backend in BACKENDS.values():
        backend.pipelines.pop(model_name, None)

    return dest


def remove_lora_file(lora_name: str, category: Optional[str] = None) -> bool:
    if category is None:
        category = _lora_category(lora_name)

    path = LORA_LOOKUP.get(lora_name)
    if category and not path:
        path = os.path.join(config.LORA_DIR, category, lora_name)

    if not path or not os.path.isfile(path):
        return False

    try:
        os.remove(path)
    finally:
        if lora_name in LORA_LOOKUP:
            del LORA_LOOKUP[lora_name]
    return True


def move_lora_file(
    lora_name: str,
    new_category: str,
    current_category: Optional[str] = None,
) -> str:
    if current_category is None:
        current_category = _lora_category(lora_name)

    src = LORA_LOOKUP.get(lora_name)
    if current_category and not src:
        src = os.path.join(config.LORA_DIR, current_category, lora_name)

    if not src or not os.path.isfile(src):
        raise FileNotFoundError(lora_name)

    dest_dir = os.path.join(config.LORA_DIR, new_category)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, lora_name)

    if os.path.abspath(src) == os.path.abspath(dest):
        return dest

    os.rename(src, dest)

    LORA_LOOKUP[lora_name] = dest
    return dest


def load_pipeline(model_name: str, category: Optional[str] = None) -> None:
    """Preload the specified model pipeline into VRAM."""
    if category is None:
        category = _model_category(model_name)
    get_pipeline(model_name, category=category)


def unload_pipeline(model_name: str | None = None, category: Optional[str] = None) -> bool:
    """Unload a model pipeline from VRAM.

    If ``model_name`` is ``None`` all loaded pipelines will be removed."""

    unloaded = False

    if model_name is not None:
        if category is None:
            category = _model_category(model_name)
        backend = get_backend(category)
        pipe = backend.pipelines.pop(model_name, None)
        if pipe:
            try:
                pipe.to("cpu")
            except Exception:
                pass
            unloaded = True
    else:
        for backend in BACKENDS.values():
            for name, pipe in list(backend.pipelines.items()):
                try:
                    pipe.to("cpu")
                except Exception:
                    pass
                del backend.pipelines[name]
                unloaded = True

    if unloaded and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return unloaded


def is_pipeline_loaded(model_name: str, category: Optional[str] = None) -> bool:
    """Return ``True`` if a pipeline is currently loaded in VRAM."""
    if category is None:
        category = _model_category(model_name)
    backend = get_backend(category)
    return model_name in backend.pipelines
