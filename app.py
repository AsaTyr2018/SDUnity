import os
import random

import gradio as gr
from PIL import Image
import numpy as np
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline

MODELS_DIR = "models"
LORA_DIR = "loras"

# Predefined HuggingFace models
PREDEFINED_MODELS = {
    # SD 1.5 models
    "sd15": "runwayml/stable-diffusion-v1-5",
    "realistic_vision_v6": "SG161222/Realistic_Vision_V6",
    "deliberate": "XpucT/Deliberate",
    "fluently_v4": "fluently/Fluently-v4",

    # SDXL models
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "fluently_xl_final": "fluently/Fluently-Xl-Final",
    "realvisxl_v4": "SG161222/RealVisXL_V4.0",
    "visionix_alpha": "ehristoforu/Visionix-alpha",
    "halcyon_1_7": "Halcyon 1.7",
    "sdxl_lightning": "SDXL-Lightning",

    # PonyXL models
    "ponyxl": "glides/ponyxl",
    "pony_diffusion_v6_xl": "LyliaEngine/Pony Diffusion V6 XL",
    "ponyxl_realistic_v3": "John6666/damn-ponyxl-realistic-v3-sdxl",
}

# Cache for loaded pipelines
PIPELINES = {}


def list_models():
    """Return available model names, including local models in MODELS_DIR."""
    names = list(PREDEFINED_MODELS.keys())
    if os.path.isdir(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            path = os.path.join(MODELS_DIR, fname)
            if os.path.isdir(path) or fname.lower().endswith((".ckpt", ".safetensors", ".bin")):
                names.append(fname)
    return names


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
            local_path = os.path.join(MODELS_DIR, model_name)
            if not os.path.exists(local_path):
                raise ValueError(f"Unknown model: {model_name}")
            repo = local_path

        if "xl" in model_name.lower() or "xl" in repo.lower():
            try:
                if os.path.isfile(repo):
                    pipe = DiffusionPipeline.from_single_file(
                        repo, torch_dtype=torch.float16, variant="fp16"
                    )
                else:
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
                if os.path.isfile(repo):
                    pipe = StableDiffusionPipeline.from_single_file(
                        repo, torch_dtype=torch.float16
                    )
                else:
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


def generate_image(
    prompt,
    negative_prompt,
    seed,
    steps,
    width,
    height,
    model,
    lora,
    nsfw_filter,
    images_per_batch,
    batch_count,
):
    """Generate one or more images using the selected diffusion model."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    pipe = get_pipeline(model)

    # Toggle safety checker based on nsfw_filter flag
    if not hasattr(pipe, "_original_safety_checker"):
        pipe._original_safety_checker = getattr(pipe, "safety_checker", None)
    pipe.safety_checker = pipe._original_safety_checker if nsfw_filter else None

    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    images = []
    for _ in range(int(batch_count)):
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            generator=generator,
            num_images_per_prompt=int(images_per_batch),
        )
        images.extend(result.images)

    if images:
        gallery_images.extend(images)
        last_img = images[-1]
    else:
        last_img = None

    return last_img, seed, gallery_images

with gr.Blocks() as demo:
    gr.Markdown("# SDUnity - Prototype")

    with gr.Row():
        with gr.Column(scale=3):
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

        with gr.Column(scale=1):
            with gr.Box():
                gr.Markdown("### Additional Settings")
                nsfw_filter = gr.Checkbox(label="NSFW Filter", value=True)
                images_per_batch = gr.Number(
                    label="Images per Batch", value=1, precision=0
                )
                batch_count = gr.Number(label="Batch Count", value=1, precision=0)

    with gr.Row():
        output = gr.Image(label="Result")
        gallery = gr.Gallery(label="Gallery")

    generate_btn.click(
        generate_image,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            steps,
            width,
            height,
            model,
            lora,
            nsfw_filter,
            images_per_batch,
            batch_count,
        ],
        outputs=[output, seed, gallery],
    )
    refresh.click(refresh_lists, outputs=[model, lora])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
