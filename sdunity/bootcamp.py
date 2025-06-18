"""Bootcamp - LoRA training utilities (WIP)."""

import os
import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, asdict, field
from typing import Generator
from html import escape

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


def _copy_image(src: str, dest_dir: str) -> str:
    """Copy ``src`` into ``dest_dir`` with a unique name."""
    name = os.path.basename(src)
    base, ext = os.path.splitext(name)
    dest = os.path.join(dest_dir, name)
    idx = 1
    while os.path.exists(dest):
        dest = os.path.join(dest_dir, f"{base}_{idx}{ext}")
        idx += 1
    shutil.copy(src, dest)
    return os.path.basename(dest)


def import_uploads(proj: BootcampProject, uploads: list[str]) -> int:
    """Import images from uploaded files or directories."""
    img_dir = os.path.join(proj.path, "images")
    os.makedirs(img_dir, exist_ok=True)
    new_imgs: list[str] = []

    for path in uploads:
        if not path:
            continue
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    fname = info.filename
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        continue
                    with zf.open(info) as src, open(
                        os.path.join(img_dir, os.path.basename(fname)), "wb"
                    ) as out:
                        shutil.copyfileobj(src, out)
                    new_imgs.append(os.path.basename(fname))
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        src_path = os.path.join(root, f)
                        new_imgs.append(_copy_image(src_path, img_dir))
        else:
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                new_imgs.append(_copy_image(path, img_dir))

    imgs = [
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    proj.images = sorted(imgs)
    for img in proj.images:
        proj.tags.setdefault(img, [])
    proj.save()
    return len(new_imgs)


def import_zip(proj: BootcampProject, zip_path: str) -> int:
    """Backward compatibility wrapper for ``import_uploads``."""
    return import_uploads(proj, [zip_path])


def tag_summary(proj: BootcampProject) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tags in proj.tags.values():
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
    return counts


def gallery_paths(proj: BootcampProject) -> list[str]:
    """Return absolute image paths for gallery display."""
    img_dir = os.path.join(proj.path, "images")
    return [os.path.join(img_dir, img).replace("\\", "/") for img in proj.images]


def render_tag_grid(proj: BootcampProject) -> str:
    """Return an HTML grid with image previews and tag buttons."""
    img_dir = os.path.join(proj.path, "images")
    html = ["<div id='bc_grid'>"]
    for img in proj.images:
        src = os.path.join(img_dir, img).replace("\\", "/")
        html.append("<div class='bc_item'>")
        html.append(f"<img src='/file={escape(src)}'/>")
        html.append("<div class='bc_tags'>")
        tags = proj.tags.get(img, [])
        if tags:
            for t in tags:
                tag = escape(t)
                html.append(
                    f"<button class='bc_tag' onclick='toggleTag(this)'>{tag}</button>"
                )
        else:
            html.append("<span class='bc_tag_none'>No Tags</span>")
        html.append("</div></div>")
    html.append("</div>")
    html.append(
        """
        <script>
        window.toggleTag = function(btn) {
            btn.classList.toggle('selected');
            const out = document.getElementById('bc_selected_tags');
            if(!out) return;
            const selected = Array.from(document.querySelectorAll('.bc_tag.selected'))
                .map(b => b.textContent.trim());
            out.value = selected.join(', ');
        }
        </script>
        """
    )
    return "\n".join(html)


def suggest_params(proj: BootcampProject, model_type: str) -> dict[str, int | float]:
    """Return simple training parameter suggestions."""
    steps = max(1000, len(proj.images) * 50)
    lr = 1e-4 if model_type != "SDXL" else 5e-5
    repeats = max(1, int(steps / max(len(proj.images), 1)))
    return {"steps": steps, "learning_rate": lr, "num_repeats": repeats}


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


def export_dataset(proj: BootcampProject, dest_zip: str) -> str:
    """Package the project images and tags into ``dest_zip``."""
    img_dir = os.path.join(proj.path, "images")
    with zipfile.ZipFile(dest_zip, "w") as zf:
        for img in proj.images:
            img_path = os.path.join(img_dir, img)
            if os.path.isfile(img_path):
                zf.write(img_path, os.path.join("images", img))
        zf.writestr("tags.json", json.dumps(proj.tags, indent=2))
    return dest_zip


def reset_project(proj: BootcampProject) -> None:
    """Remove all dataset files from the project and reset metadata."""
    shutil.rmtree(os.path.join(proj.path, "images"), ignore_errors=True)
    proj.images = []
    proj.tags = {}
    proj.save()
