"""Lightweight package loader for SDUnity submodules."""

from importlib import import_module

__all__ = [
    "presets",
    "model_loader",
    "models",
    "gallery",
    "generator",
    "config",
    "civitai",
    "tags",
    "wildcards",
    "settings_presets",
]


def __getattr__(name: str):
    """Dynamically import submodules on first access.

    This avoids importing heavy dependencies like ``torch`` and
    ``diffusers`` unless the corresponding modules are actually used.
    """
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
