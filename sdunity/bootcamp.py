"""Bootcamp - LoRA training utilities (WIP)."""

import shutil
import subprocess
from typing import Generator

import gradio as gr


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
