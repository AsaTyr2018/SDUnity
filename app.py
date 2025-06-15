import os
import random

import gradio as gr
from PIL import Image
import numpy as np
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline

MODELS_DIR = "models"
LORA_DIR = "loras"

# Predefined huggingface models
PREDEFINED_MODELS = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "ponyxl": "glides/ponyxl",
}

# Cache for loaded pipelines
PIPELINES = {}


def list_models():
    """Return available model names."""
    return list(PREDEFINED_MODELS.keys())


def list_loras():
    if not os.path.isdir(LORA_DIR):
        return []
    return [f for f in os.listdir(LORA_DIR) if f.lower().endswith(".safetensors")]


def refresh_lists():
    """Return updated choices for model and LoRA dropdowns."""
    return gr.update(choices=list_models()), gr.update(choices=list_loras())


gallery_images = []

def get_pipeline(model_name):
    """Load and cache the diffusion pipeline for the given model."""
    if model_name not in PIPELINES:
        repo = PREDEFINED_MODELS.get(model_name)
        if repo is None:
            raise ValueError(f"Unknown model: {model_name}")
        if model_name == "sdxl":
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    repo, torch_dtype=torch.float16, variant="fp16"
                )
            except ImportError as e:
                raise RuntimeError(
                    "StableDiffusionXLPipeline requires the 'transformers' library. "
                    "Install it with `pip install transformers`."
                ) from e
        else:
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    repo, torch_dtype=torch.float16
                )
            except ImportError as e:
                raise RuntimeError(
                    "StableDiffusionPipeline requires the 'transformers' library. "
                    "Install it with `pip install transformers`."
                ) from e
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        PIPELINES[model_name] = pipe
    return PIPELINES[model_name]


def generate_image(prompt, negative_prompt, seed, steps, width, height, model, lora):
    """Generate an image using the selected diffusion model."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    pipe = get_pipeline(model)
    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=int(width),
        height=int(height),
        num_inference_steps=int(steps),
        generator=generator,
    )
    img = result.images[0]
    gallery_images.append(img)
    return img, seed, gallery_images

with gr.Blocks() as demo:
    gr.Markdown("# SDUnity - Prototype")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
        negative_prompt = gr.Textbox(label="Negative Prompt")

    with gr.Row():
        seed = gr.Number(label="Seed", value=None, precision=0)
        steps = gr.Slider(1, 50, value=20, label="Steps")
        width = gr.Slider(64, 1024, value=256, step=64, label="Width")
        height = gr.Slider(64, 1024, value=256, step=64, label="Height")

    with gr.Row():
        model = gr.Dropdown(choices=list_models(), value="sd15", label="Model")
        lora = gr.Dropdown(choices=list_loras(), label="LoRA", multiselect=True)
        refresh = gr.Button("Refresh")

    generate_btn = gr.Button("Generate")
    with gr.Row():
        output = gr.Image(label="Result")
        gallery = gr.Gallery(label="Gallery")

    generate_btn.click(
        generate_image,
        inputs=[prompt, negative_prompt, seed, steps, width, height, model, lora],
        outputs=[output, seed, gallery],
    )
    refresh.click(refresh_lists, outputs=[model, lora])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
