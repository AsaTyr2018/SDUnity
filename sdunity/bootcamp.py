"""Bootcamp - LoRA training utilities (WIP)."""

import os
import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, asdict, field
from typing import Generator

import gradio as gr

from . import config


@dataclass
class BootcampProject:
    """Basic representation of a Bootcamp training project."""

    name: str
    lora_type: str
    path: str
    images: list[str] = field(default_factory=list)
    tags: dict[str, list[str]] = field(default_factory=dict)
    prompts: list[str] = field(
        default_factory=lambda: ["Automatically set", "", ""]
    )

    def save(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "project.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(name: str) -> "BootcampProject | None":
        path = os.path.join(config.BOOTCAMP_PROJECTS_DIR, name)
        meta = os.path.join(path, "project.json")
        if not os.path.isfile(meta):
            return None
        with open(meta, "r", encoding="utf-8") as f:
            data = json.load(f)
        return BootcampProject(**data)


def create_project(name: str, lora_type: str) -> BootcampProject:
    path = os.path.join(config.BOOTCAMP_PROJECTS_DIR, name)
    proj = BootcampProject(name=name, lora_type=lora_type, path=path)
    proj.save()
    return proj


def import_zip(proj: BootcampProject, zip_path: str) -> int:
    """Extract images from ``zip_path`` into the project directory."""
    img_dir = os.path.join(proj.path, "images")
    os.makedirs(img_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(img_dir)
    imgs = [
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    proj.images = imgs
    for img in imgs:
        proj.tags.setdefault(img, [])
    proj.save()
    return len(imgs)


def tag_summary(proj: BootcampProject) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tags in proj.tags.values():
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
    return counts


def suggest_params(proj: BootcampProject, model_type: str) -> dict[str, int | float]:
    """Return simple training parameter suggestions."""
    steps = max(1000, len(proj.images) * 50)
    lr = 1e-4 if model_type != "SDXL" else 5e-5
    return {"steps": steps, "learning_rate": lr}


def run_training(
    proj: BootcampProject, model: str, steps: int, learning_rate: float
) -> Generator[str, None, None]:
    out_dir = os.path.join(config.BOOTCAMP_OUTPUT_DIR, proj.name)
    os.makedirs(out_dir, exist_ok=True)
    inst_dir = os.path.join(proj.path, "images")
    yield from train_lora(
        inst_dir,
        model,
        out_dir,
        steps=steps,
        learning_rate=learning_rate,
    )


BOOTCAMP_SCRIPT = "train_dreambooth_lora.py"


def train_lora(
    instance_dir: str,
    pretrained_model: str,
    output_dir: str,
    steps: int = 1000,
    learning_rate: float = 1e-4,
    progress: gr.Progress | None = None,
) -> Generator[str, None, None]:
    """Launch LoRA training via the diffusers DreamBooth script.

    This is a minimal wrapper around the official training script and is
    provided as a work in progress. It requires ``accelerate`` and
    ``diffusers[training]`` to be installed and accessible on the system.
    """
    if progress:
        progress(0, desc="Bootcamp initialising")

    if not shutil.which("accelerate"):
        yield "`accelerate` command not found. Please install diffusers training requirements."
        return

    cmd = [
        "accelerate",
        "launch",
        BOOTCAMP_SCRIPT,
        "--pretrained_model_name_or_path",
        pretrained_model,
        "--instance_data_dir",
        instance_dir,
        "--output_dir",
        output_dir,
        "--max_train_steps",
        str(int(steps)),
        "--learning_rate",
        str(learning_rate),
    ]
    yield f"Starting Bootcamp with {steps} steps..."
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        yield f"Training failed: {exc}"
        return

    yield f"Training complete. LoRA saved to {output_dir}"
